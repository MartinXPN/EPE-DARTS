""" Operations """
import torch
import torch.nn as nn

from epe_darts import genotypes as gt

OPS = {
    'none': lambda channels, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda channels, stride, affine: PoolBN('avg', channels, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda channels, stride, affine: PoolBN('max', channels, 3, stride, 1, affine=affine),
    'skip_connect': lambda channels, stride, affine: Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=affine),
    'sep_conv_3x3': lambda channels, stride, affine: SepConv(channels, channels, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda channels, stride, affine: SepConv(channels, channels, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda channels, stride, affine: SepConv(channels, channels, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda channels, stride, affine: DilConv(channels, channels, 3, stride, 2, 2, affine=affine),  # 5x5
    'dil_conv_5x5': lambda channels, stride, affine: DilConv(channels, channels, 5, stride, 4, 2, affine=affine),  # 9x9
    'conv_7x1_1x7': lambda channels, stride, affine: FacConv(channels, channels, 7, stride, 3, affine=affine),
    'nor_conv_7x7': lambda channels, stride, affine: ReLUConvBN(channels, channels, 7, stride, 3, affine=affine),
    'nor_conv_3x3': lambda channels, stride, affine: ReLUConvBN(channels, channels, 3, stride, 1, affine=affine),
    'nor_conv_1x1': lambda channels, stride, affine: ReLUConvBN(channels, channels, 1, stride, 0, affine=affine),
}


CONNECT_NAS_BENCHMARK = ['none', 'skip_connect', 'nor_conv_3x3']
NAS_BENCH_201         = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
DARTS                 = ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3',
                         'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3']
SEARCH_SPACE2OPS = {
    'connect-nas-bench': CONNECT_NAS_BENCHMARK,
    'nas-bench-201': NAS_BENCH_201,
    'darts': DARTS,
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, channels, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(channels, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, in_channels, out_channels, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(in_channels, out_channels, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(in_channels, in_channels, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(in_channels, out_channels, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class ReLUConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=1, bias=not affine),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, in_channels, out_channels, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, channels, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](channels, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))
