import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr import build_model
from basicsr.archs.NAFNet_arch import NAFNet, NAFNetLocal
from basicsr.archs.safmn_fft_arch import SAFMN_FFT
from raw_kit.load_data import load_data
from basicsr.data.transforms import augment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='../models/SAFMN_FFT.pth', help='model path'
    )
    parser.add_argument('--input', type=str, default='D:/Datasets/DSLR/val_dev/val_in', help='input test image folder')
    parser.add_argument('--output', type=str, default='D:/Datasets/DSLR/val_dev/val_pred',
                        help='output folder')
    # parser.add_argument('--local', action='store_true')
    parser.add_argument('--self_ensemble', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = SAFMN_FFT(
        in_chans=4,
        dim=36,
        n_blocks=8,
        ffn_scale=2.0,
        upscaling_factor=2
    )
    # model = NAFNetLocal(
    #     width=64,
    #     enc_blk_nums=[2, 2, 4, 8],
    #     middle_blk_num=12,
    #     dec_blk_nums=[2, 2, 2, 2],
    #     img_channel=4,
    #     upscale=2,
    #     nafb_g=1,
    #     train_size=(1, 4, 128, 128)
    # )
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    print('loaded parameters from {}'.format(args.model_path))
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    if args.self_ensemble:
        print('using self ensemble')
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img, maxval = load_data(path)
        print(path, img.shape)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0).to(device)

        if not args.self_ensemble:
            # inference
            try:
                with torch.no_grad():
                    output = model(img)
            except Exception as error:
                print('Error', error, imgname)
            else:
                # save image
                output = output.data.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                # print(output.max(), output.min())
                output_img = (output * maxval).round().astype(np.uint16)

                '''save'''
                np.savez(os.path.join(args.output, f'{imgname}.npz'), raw=output_img, max_val=maxval)
        else:

            # inference
            # 8 augmentations
            # modified from https://github.com/thstkdgus35/EDSR-PyTorch
            def _transform(v, op):
                # if self.precision != 'single': v = v.float()
                v2np = v.data.cpu().numpy()
                if op == 'v':
                    tfnp = v2np[:, :, :, ::-1].copy()
                elif op == 'h':
                    tfnp = v2np[:, :, ::-1, :].copy()
                elif op == 't':
                    tfnp = v2np.transpose((0, 1, 3, 2)).copy()

                ret = torch.Tensor(tfnp).to(device)
                # if self.precision == 'half': ret = ret.half()
                return ret

            # prepare augmented data
            lq_list = [img]
            for tf in 'v', 'h', 't':
                lq_list.extend([_transform(t, tf) for t in lq_list])
            # inference
            with torch.no_grad():
                out_list = [model(aug) for aug in lq_list]
            # merge results
            for i in range(len(out_list)):
                if i > 3:
                    out_list[i] = _transform(out_list[i], 't')
                if i % 4 > 1:
                    out_list[i] = _transform(out_list[i], 'h')
                if (i % 4) % 2 == 1:
                    out_list[i] = _transform(out_list[i], 'v')
            output = torch.cat(out_list, dim=0)
            output = output.mean(dim=0, keepdim=True)
            output = output.data.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            print(output.max(), output.min())
            output_img = (output * maxval).round().astype(np.uint16)
            np.savez(os.path.join(args.output, f'{imgname}.npz'), raw=output_img, max_val=maxval)


if __name__ == '__main__':
    main()
