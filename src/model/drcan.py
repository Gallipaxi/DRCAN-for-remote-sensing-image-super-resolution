import torch
import math
from model import common



import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return DRCAN(args)



class DRCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DRCAN, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        m_beforehead = [conv(args.n_colors, n_feats, kernel_size)]

        head_n_blocks = 6
        # head_n_blocks = 10
        m_head = []
        for _ in range(head_n_blocks):
            m_head.append(common.WDSR_ATT(conv, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1, stride=1, downsample=None))

        m_body = []
        # body_n1_blocks = 20
        body_n1_blocks = 4
        body_n2_blocks = 4

        body_block =[]
        body_conv_channel = int(n_feats/(scale**2))  # body_conv_channel = 16

        # define body_1
        for _ in range(body_n1_blocks):
            body_block.append(
                common.LR_block(conv, n_feats, kernel_size, wn, act, res_scale=1)
            )
        m_body.append(body_block)

       # define body_2
        big_block = []
        for _ in range(body_n2_blocks):
            big_block.append(common.HR_block(n_feats))

        if (scale & (scale-1)) == 0:
            block = []
            block.append(common.Upsampler(conv, scale, n_feats, act=False))
            block.extend(big_block)
            m_body.append(block)

        m_tail = []
        # n_tail = 4
        n_tail = 1


        for _ in range(n_tail):
            m_tail.append(common.WDSR(conv, n_feats, kernel_size, wn=wn, act=act, res_scale=args.res_scale))

        m_tail.append(conv(n_feats, args.n_colors, kernel_size))

        self.beforehead = nn.Sequential(*m_beforehead)
        self.head = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body[0])
        self.body2 = nn.Sequential(*m_body[1])

        self.tail = nn.Sequential(*m_tail)
        self.default_deconv = nn.ConvTranspose2d(in_channels=n_feats, out_channels=n_feats,
                                                 kernel_size=(4,4), stride=2, padding=1, output_padding=0, bias=False)
        self.pixelUP = common.Upsampler(conv, scale, n_feats, act=False)
        self.conv = common.conv3x3(n_feats, n_feats)
        self.conv2_1 = common.conv3x3(2*n_feats, n_feats)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.beforehead(x)
        ori_res = x
        x = self.head(x)
        x = self.conv(x)
        x = ori_res + x

        x1 = self.body1(x)
        x1 = self.conv(x1)
        x1 = x1 + x
        x1 = self.conv(x1)
        x1 = x1 + ori_res
        x1 = self.pixelUP(x1)

        x2 = self.body2(x)

        x = torch.cat([x2, x1], 1)
        x = self.conv2_1(x)
        x = self.tail(x)
        # x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
