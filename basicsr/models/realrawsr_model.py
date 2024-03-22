import torch
import torch.nn.functional as F
import numpy as np
import random
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from raw_kit.blur import apply_psf
from raw_kit.imutils import downsample_raw
from raw_kit.noise import add_natural_noise, add_heteroscedastic_gnoise
from .realesrnet_model import RealESRNetModel
from .sr_model import SRModel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
import os.path as osp
from tqdm import tqdm

from ..metrics import calculate_metric
from raw_kit.degradations import add_blur, add_heteroscedastic_gnoise, add_natural_noise, linear_exposure_compensation


@MODEL_REGISTRY.register()
class RealRawSRModel(SRModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(RealRawSRModel, self).__init__(opt)
        self.queue_size = opt.get('queue_size', 180)
        self.raw_kernels = np.load(opt.get('raw_kernel_path'), allow_pickle=True)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
             """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)
            self.max_value = data['max_value']
            ori_h, ori_w = self.gt.size()[2:4]

            # blur
            psf_prob1 = np.random.uniform()
            if psf_prob1 > self.opt['psf_apply_prob1']:
                out = filter2D(self.gt, self.kernel1)
            else:
                out = add_blur(self.gt, kernels=self.raw_kernels)

            # linear exposure
            exposure_prob = np.random.rand()
            if exposure_prob < self.opt['exposure_prob']:
                compensation_value = np.random.uniform(self.opt['exposure_compensation_range'][0],
                                                       self.opt['exposure_compensation_range'][1])
                out = linear_exposure_compensation(out, 1, compensation_value)

            # downsample
            out = downsample_raw(out)

            # # random resize
            # updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            # if updown_type == 'up':
            #     scale = np.random.uniform(1, self.opt['resize_range'][1])
            # elif updown_type == 'down':
            #     scale = np.random.uniform(self.opt['resize_range'][0], 1)
            # else:
            #     scale = 1
            # mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(out, scale_factor=scale, mode=mode)
            # # resize back
            # mode = random.choice(['area', 'bilinear', 'bicubic'])
            # out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
            #
            # # clamp to 0~1
            # out = torch.clamp(out, 0, 1)

            # add noise
            p_noise = np.random.uniform() - exposure_prob * 0.5  # if darker, prefer shot-read noise
            if p_noise < self.opt['sr_noise_prob1']:
                out = add_natural_noise(out, self.device)  # shot-read noise
            else:
                out = add_heteroscedastic_gnoise(out, self.device, self.opt['sigma_1_range'],
                                                 self.opt['sigma_2_range'])  # heteroscedastic gaussian
            '''
            # second blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            '''
            # # sinc blur
            # out = filter2D(out, self.sinc_kernel)

            # clamp and round
            for idx in range(out.size()[0]):
                out[idx] = torch.clamp((out[idx] * self.max_value[idx]).round(), 0, self.max_value[idx]) / \
                           self.max_value[idx]

            # import cv2
            # cv2.imshow('in', self.gt[0].cpu().permute(1, 2, 0).detach().numpy())
            # cv2.imshow('out', out[0].cpu().permute(1, 2, 0).detach().numpy())
            # cv2.waitKey(0)

            self.lq = out
            # random crop
            gt_size = self.opt['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super().nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def nondist_validation_selfensemble(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test_selfensemble()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.is_train = True
