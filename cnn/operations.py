import torch
import torch.nn as nn

# OPS is a set of layers with same input/output channel.

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):
    """
    Stack of relu-conv-bn
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding:
        :param affine:
        """
        super(ReLUConvBN, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """
    relu-dilated conv-bn
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding: 2/4
        :param dilation: 2
        :param affine:
        """
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),  # 1 x 1 的卷积核用来降低通道数
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """ 深度可分离卷积 separate Convolution
    implemented separate convolution via torch groups parameters
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        """

        :param C_in:
        :param C_out:
        :param kernel_size:
        :param stride:
        :param padding: 1/2
        :param affine:
        """
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """
    zero by stride
    """

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    """ 是利用stride = 2来进行减少高度和宽度操作的
    reduce feature maps height/width by half while keeping channel same
    """

    def __init__(self, C_in, C_out, affine=True):
        """

        :param C_in:
        :param C_out:
        :param affine:
        """
        super().__init__()

        assert C_out % 2 == 0

        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        # x: torch.Size([32, 32, 32, 32])
        # conv1: [b, c_out//2, d//2, d//2]
        # conv2: []
        # out: torch.Size([32, 32, 16, 16])
        tmp_x1 = self.conv_1(x)
        tmp_x2 = self.conv_2(x[:, :, 1:, 1:])
        out = torch.cat([tmp_x1, tmp_x2], dim=1)
        out = self.bn(out)
        return out
