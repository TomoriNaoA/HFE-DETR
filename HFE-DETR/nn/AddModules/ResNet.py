
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
 
# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#     default_act = nn.SiLU()  # default activation
 
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))
 
#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))

# class Channel(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dwconv = self.dconv = nn.Conv2d(
#             dim, dim, 3,
#             1, 1, groups=dim
#         )
#         self.Apt = nn.AdaptiveAvgPool2d(1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x2 = self.dwconv(x)
#         x5 = self.Apt(x2)
#         x6 = self.sigmoid(x5)

#         return x6


# class Spatial(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(dim, 1, 1, 1)
#         self.bn = nn.BatchNorm2d(1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x5 = self.bn(x1)
#         x6 = self.sigmoid(x5)

#         return x6

# class FCM(nn.Module):
#     def __init__(self, dim,dim_out):
#         super().__init__()
#         self.one = dim // 4
#         self.two = dim - dim // 4
#         self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
#         self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
#         self.conv123 = Conv(dim // 4, dim, 1, 1)

#         self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
#         self.conv3 = Conv(dim, dim, 1, 1)
#         self.spatial = Spatial(dim)
#         self.channel = Channel(dim)

#     def forward(self, x):
#         x1, x2 = torch.split(x, [self.one, self.two], dim=1)
#         x3 = self.conv1(x1)
#         x3 = self.conv12(x3)
#         x3 = self.conv123(x3)
#         x4 = self.conv2(x2)
#         x33 = self.spatial(x4) * x3
#         x44 = self.channel(x3) * x4
#         x5 = x33 + x44
#         x5 = self.conv3(x5)
#         return x5

# class ConvNormLayer(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  filter_size,
#                  stride,
#                  groups=1,
#                  act=None):
#         super(ConvNormLayer, self).__init__()
#         self.act = act
#         self.conv = nn.Conv2d(
#             in_channels=ch_in,
#             out_channels=ch_out,
#             kernel_size=filter_size,
#             stride=stride,
#             padding=(filter_size - 1) // 2,
#             groups=groups)
 
#         self.norm = nn.BatchNorm2d(ch_out)
 
#     def forward(self, inputs):
#         out = self.conv(inputs)
#         out = self.norm(out)
#         if self.act:
#             out = getattr(F, self.act)(out)
#         return out
 
# class SELayer(nn.Module):
#     def __init__(self, ch, reduction_ratio=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(ch, ch // reduction_ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(ch // reduction_ratio, ch, bias=False),
#             nn.Sigmoid()
#         )
 
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class BasicBlock_FCM(nn.Module):
#     expansion = 1
 
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  stride,
#                  shortcut,
#                  act='relu',
#                  variant='b',
#                  att=False):
#         super(BasicBlock_FCM, self).__init__()
#         self.shortcut = shortcut
#         if not shortcut:
#             if variant == 'd' and stride == 2:
#                 self.short = nn.Sequential()
#                 self.short.add_sublayer(
#                     'pool',
#                     nn.AvgPool2d(
#                         kernel_size=2, stride=2, padding=0, ceil_mode=True))
#                 self.short.add_sublayer(
#                     'conv',
#                     ConvNormLayer(
#                         ch_in=ch_in,
#                         ch_out=ch_out,
#                         filter_size=1,
#                         stride=1))
#             else:
#                 self.short = ConvNormLayer(
#                     ch_in=ch_in,
#                     ch_out=ch_out,
#                     filter_size=1,
#                     stride=stride)
 
#         self.branch2a = ConvNormLayer(
#             ch_in=ch_in,
#             ch_out=ch_out,
#             filter_size=3,
#             stride=stride,
#             act='relu')
 
#         self.branch2b = ConvNormLayer(
#             ch_in=ch_out,
#             ch_out=ch_out,
#             filter_size=3,
#             stride=1,
#             act=None)
 
#         self.att = att
#         if self.att:
#             self.se = FCM(ch_out, ch_out)
 
#     def forward(self, inputs):
#         out = self.branch2a(inputs)
#         out = self.branch2b(out)
 
#         if self.att:
#             out = self.se(out)
 
#         if self.shortcut:
#             short = inputs
#         else:
#             short = self.short(inputs)
 
#         out = out + short
#         out = F.relu(out)
 
#         return out
 
# class BottleNeck(nn.Module):
#     expansion = 4
 
#     def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', att=False):
#         super().__init__()
 
#         if variant == 'a':
#             stride1, stride2 = stride, 1
#         else:
#             stride1, stride2 = 1, stride
 
#         width = ch_out
 
#         self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
#         self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
#         self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
 
#         self.shortcut = shortcut
#         if not shortcut:
#             if variant == 'd' and stride == 2:
#                 self.short = nn.Sequential(OrderedDict([
#                     ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
#                     ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
#                 ]))
#             else:
#                 self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
 
#         self.att = att
#         if self.att:
#             self.se = SELayer(ch_out * 4)
 
#     def forward(self, x):
#         out = self.branch2a(x)
#         out = self.branch2b(out)
#         out = self.branch2c(out)
 
#         if self.att:
#             out = self.se(out)
 
#         if self.shortcut:
#             short = x
#         else:
#             short = self.short(x)
 
#         out = out + short
#         out = F.relu(out)
 
#         return out
 
# class Blocks(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  count,
#                  block,
#                  stage_num,
#                  att=False,
#                  variant='b'):
#         super(Blocks, self).__init__()
#         self.blocks = nn.ModuleList()
#         block = globals()[block]
#         for i in range(count):
#             self.blocks.append(
#                 block(
#                     ch_in,
#                     ch_out,
#                     stride=2 if i == 0 and stage_num != 2 else 1,
#                     shortcut=False if i == 0 else True,
#                     variant=variant,
#                     att=att)
#             )
#             if i == 0:
#                 ch_in = ch_out * block.expansion
 
#     def forward(self, inputs):
#         block_out = inputs
#         for block in self.blocks:
#             block_out = block(block_out)
#         return block_out


'''resnet18-ConvLayers'''

from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 act=None):
        super(ConvNormLayer, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups)

        self.norm = nn.BatchNorm2d(ch_out)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        if self.act:
            out = getattr(F, self.act)(out)
        return out

class SELayer(nn.Module):
    def __init__(self, ch, reduction_ratio=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction_ratio, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 ch_in,
                 ch_out,
                 stride,
                 shortcut,
                 act='relu',
                 variant='b',
                 att=False):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.need_downsample = (stride == 2)

        # =============== 修改 shortcut 分支 ===============
        if not shortcut:
            if self.need_downsample:
                # 使用 SPDConv 替代下采样
                self.short = SPDConv(ch_in, ch_out, k=1, s=1, p=0, act=False)
            else:
                self.short = ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=1,
                    stride=1)  # stride=1

        # =============== 修改主分支 ===============
        if self.need_downsample:
            # 主分支第一个卷积用 SPDConv（代替 stride=2）
            self.branch2a = SPDConv(ch_in, ch_out, k=3, s=1, p=1, act='relu')
        else:
            self.branch2a = ConvNormLayer(
                ch_in=ch_in,
                ch_out=ch_out,
                filter_size=3,
                stride=1,
                act='relu')

        self.branch2b = ConvNormLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            act=None)

        self.att = att
        if self.att:
            self.se = SELayer(ch_out)

    def forward(self, inputs):
        out = self.branch2a(inputs)
        out = self.branch2b(out)

        if self.att:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = out + short
        out = F.relu(out)
        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', att=False):
        super().__init__()
        self.need_downsample = (stride == 2)

        # =============== 主分支 ===============
        if self.need_downsample:
            # 第一个卷积用 SPDConv
            self.branch2a = SPDConv(ch_in, ch_out, k=1, s=1, p=0, act=act)
        else:
            self.branch2a = ConvNormLayer(ch_in, ch_out, 1, 1, act=act)

        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=act)  # 中间卷积 stride=1
        self.branch2c = ConvNormLayer(ch_out, ch_out * self.expansion, 1, 1)

        # =============== shortcut 分支 ===============
        self.shortcut = shortcut
        if not shortcut:
            if self.need_downsample:
                self.short = SPDConv(ch_in, ch_out * self.expansion, k=1, s=1, p=0, act=False)
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)

        self.att = att
        if self.att:
            self.se = SELayer(ch_out * 4)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.att:
            out = self.se(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = F.relu(out)
        return out

class Blocks(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 count,
                 block,
                 stage_num,
                 att=False,
                 variant='b'):
        super(Blocks, self).__init__()
        self.blocks = nn.ModuleList()
        block = globals()[block]
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    att=att)
            )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out



import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))


'''resnet18Layers'''


# from collections import OrderedDict
# import torch.nn as nn
# import torch.nn.functional as F


# class ConvNormLayer(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  filter_size,
#                  stride,
#                  groups=1,
#                  act=None):
#         super(ConvNormLayer, self).__init__()
#         self.act = act
#         self.conv = nn.Conv2d(
#             in_channels=ch_in,
#             out_channels=ch_out,
#             kernel_size=filter_size,
#             stride=stride,
#             padding=(filter_size - 1) // 2,
#             groups=groups)

#         self.norm = nn.BatchNorm2d(ch_out)

#     def forward(self, inputs):
#         out = self.conv(inputs)
#         out = self.norm(out)
#         if self.act:
#             out = getattr(F, self.act)(out)
#         return out


# class SELayer(nn.Module):
#     def __init__(self, ch, reduction_ratio=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(ch, ch // reduction_ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(ch // reduction_ratio, ch, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  stride,
#                  shortcut,
#                  act='relu',
#                  variant='b',
#                  att=False):
#         super(BasicBlock, self).__init__()
#         self.shortcut = shortcut
#         if not shortcut:
#             if variant == 'd' and stride == 2:
#                 self.short = nn.Sequential()
#                 self.short.add_sublayer(
#                     'pool',
#                     nn.AvgPool2d(
#                         kernel_size=2, stride=2, padding=0, ceil_mode=True))
#                 self.short.add_sublayer(
#                     'conv',
#                     ConvNormLayer(
#                         ch_in=ch_in,
#                         ch_out=ch_out,
#                         filter_size=1,
#                         stride=1))
#             else:
#                 self.short = ConvNormLayer(
#                     ch_in=ch_in,
#                     ch_out=ch_out,
#                     filter_size=1,
#                     stride=stride)

#         self.branch2a = ConvNormLayer(
#             ch_in=ch_in,
#             ch_out=ch_out,
#             filter_size=3,
#             stride=stride,
#             act='relu')

#         self.branch2b = ConvNormLayer(
#             ch_in=ch_out,
#             ch_out=ch_out,
#             filter_size=3,
#             stride=1,
#             act=None)

#         self.att = att
#         if self.att:
#             self.se = SELayer(ch_out)

#     def forward(self, inputs):
#         out = self.branch2a(inputs)
#         out = self.branch2b(out)

#         if self.att:
#             out = self.se(out)

#         if self.shortcut:
#             short = inputs
#         else:
#             short = self.short(inputs)

#         out = out + short
#         out = F.relu(out)

#         return out


# class BottleNeck(nn.Module):
#     expansion = 4

#     def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d', att=False):
#         super().__init__()

#         if variant == 'a':
#             stride1, stride2 = stride, 1
#         else:
#             stride1, stride2 = 1, stride

#         width = ch_out

#         self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
#         self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
#         self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

#         self.shortcut = shortcut
#         if not shortcut:
#             if variant == 'd' and stride == 2:
#                 self.short = nn.Sequential(OrderedDict([
#                     ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
#                     ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
#                 ]))
#             else:
#                 self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

#         self.att = att
#         if self.att:
#             self.se = SELayer(ch_out * 4)

#     def forward(self, x):
#         out = self.branch2a(x)
#         out = self.branch2b(out)
#         out = self.branch2c(out)

#         if self.att:
#             out = self.se(out)

#         if self.shortcut:
#             short = x
#         else:
#             short = self.short(x)

#         out = out + short
#         out = F.relu(out)

#         return out


# class Blocks(nn.Module):
#     def __init__(self,
#                  ch_in,
#                  ch_out,
#                  count,
#                  block,
#                  stage_num,
#                  att=False,
#                  variant='b'):
#         super(Blocks, self).__init__()
#         self.blocks = nn.ModuleList()
#         block = globals()[block]
#         for i in range(count):
#             self.blocks.append(
#                 block(
#                     ch_in,
#                     ch_out,
#                     stride=2 if i == 0 and stage_num != 2 else 1,
#                     shortcut=False if i == 0 else True,
#                     variant=variant,
#                     att=att)
#             )
#             if i == 0:
#                 ch_in = ch_out * block.expansion

#     def forward(self, inputs):
#         block_out = inputs
#         for block in self.blocks:
#             block_out = block(block_out)
#         return block_out
