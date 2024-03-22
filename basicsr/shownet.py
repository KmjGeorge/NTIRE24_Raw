from os import path as osp

from basicsr.models import build_model
from basicsr.utils.options import parse_options
from ptflops import get_model_complexity_info
from torchsummary import summary
import torch
from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis


def shownet(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    model = build_model(opt)
    model_g = model.net_g
    input_size = int(opt['datasets']['train']['gt_size'] / opt['scale'])
    x = torch.randn(1, 4, input_size, input_size).cuda()

    summary(model_g, input_size=(4, input_size, input_size))
    print('\n\n==================================fvcore statics===============================')
    print(flop_count_table(FlopCountAnalysis(model_g, x), activations=ActivationCountAnalysis(model_g, x)))


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    shownet(root_path)
