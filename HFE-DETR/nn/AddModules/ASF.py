import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))
 
 
class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)
 
    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # l = self.conv_l_post_down(l)
        # m = self.conv_m(m)
        # s = self.conv_s_pre_up(s)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        # s = self.conv_s_post_up(s)
        lms = torch.cat([l, m, s], dim=1)
        return lms
 
 
class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()

    #     # 原始的通道对齐
    #     self.conv0 = Conv(inc[0], channel, 1)
    #     self.conv1 = Conv(inc[1], channel, 1)
    #     self.conv2 = Conv(inc[2], channel, 1)
    #     self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
    #     self.bn = nn.BatchNorm3d(channel)
    #     self.act = nn.LeakyReLU(0.1)
    #     self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))
    
    # def forward(self, x):
    #     p3, p4, p5 = x[0], x[1], x[2]
    #     p3 = self.conv0(p3)
    #     p4_2 = self.conv1(p4)
    #     p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
    #     p5_2 = self.conv2(p5)
    #     p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
    #     p3_3d = torch.unsqueeze(p3, -3)
    #     p4_3d = torch.unsqueeze(p4_2, -3)
    #     p5_3d = torch.unsqueeze(p5_2, -3)
    #     combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
    #     conv_3d = self.conv3d(combine)
    #     bn = self.bn(conv_3d)
    #     act = self.act(bn)
    #     x = self.pool_3d(act)
    #     x = torch.squeeze(x, 2)
    #     return x


    
        # 改进后的通道对齐
        self.p3_conv = Conv(inc[0], channel, 1)  # p3: low-level feature
        self.p4_conv = Conv(inc[1], channel, 1)  # p4: mid-level feature
        self.p5_conv = Conv(inc[2], channel, 1)  # p5: high-level feature

        # Step 2: 不对称卷积增强局部细节
        self.p3_asym = nn.Sequential(
            Conv(channel, channel, (1, 3), 1, (0, 1)),  # 1×3 卷积
            Conv(channel, channel, (3, 1), 1, (1, 0))   # 3×1 卷积
        )
        self.p4_asym = nn.Sequential(
            Conv(channel, channel, (1, 3), 1, (0, 1)),
            Conv(channel, channel, (3, 1), 1, (1, 0))
        )

    def forward(self, x):
        p3, p4, p5 = x  # 输入为 [p3, p4, p5]

        # Step 1: 通道对齐
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # Step 2: 上采样 p4, p5 到 p3 尺寸
        p4_up = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
        p5_up = F.interpolate(p5, size=p3.shape[2:], mode='nearest')

        # Step 3: 跨尺度融合
        # p3 与 p4, p5 特征融合
        p3_fused = p3 + p4_up + p5_up
        # p3_fused = self.p3_asym(p3_fused)  # 不对称卷积增强细节

        return p3_fused  # 返回增强后的 p3
 
 
class Add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
 
    def forward(self, x):
        input1, input2 = x[0], x[1]
        x = input1 + input2
        return x
 
'''CAPM'''
# class channel_att(nn.Module):
#     def __init__(self, channel, b=1, gamma=2):
#         super(channel_att, self).__init__()
#         kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = y.squeeze(-1)
#         y = y.transpose(-1, -2)
#         y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)



'''DyCAPM'''

class channel_att(nn.Module):
    def __init__(self, channel, reduction=4, b=1, gamma=2):
        super(channel_att, self).__init__()
        # 动态评分头：轻量 MLP 生成通道重要性
        self.score_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

        # Heavy 路径：自适应大卷积核（与原 channel_att 一致）
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv_heavy = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # Light 路径：1x1 卷积
        self.conv_light = nn.Conv1d(1, 1, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 通道重要性评分 [B, C, 1, 1]
        scores = self.score_net(x)  # [B, C, 1, 1]
        scores = scores.view(B, C)  # [B, C]

        # GAP for attention input: [B, 1, C]
        y = F.adaptive_avg_pool2d(x, 1).view(B, 1, C)

        # 双路径处理
        y_heavy = self.conv_heavy(y)   # [B, 1, C]
        y_light = self.conv_light(y)   # [B, 1, C]

        # 动态软融合：高分通道更依赖 heavy 路径
        alpha = scores.unsqueeze(1)  # [B, 1, C]
        y_fused = alpha * y_heavy + (1.0 - alpha) * y_light  # [B, 1, C]

        # 激活并扩展回原 shape
        y_fused = self.sigmoid(y_fused).view(B, C, 1, 1)
        return x * y_fused.expand_as(x)



class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()
 
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
 
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
 
 
class attention_model(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, ch=256):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)
 
    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x

 




 


 
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
 
 


