import torch
import torch.nn as nn
import torch.nn.functional as F

# import fusion_strategy


class ConvLayer(nn.Module):
    def __init__(self, inp, outp, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        self.is_last = is_last

        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(inp, outp, kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode='reflect')

    def forward(self, x):
        out = self.conv2d(x)
        if not self.is_last:
            out = F.relu(out, inplace=True)
            # out = f.dropout2d(out, training=self.training, p=0.1)
        return out


# Dense convolution unit
class DenseLayer(nn.Module):
    def __init__(self, inp, outp, kernel_size, stride):
        super(DenseLayer, self).__init__()
        self.dense_conv = ConvLayer(inp, outp, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(nn.Module):
    def __init__(self, inp, kernel_size, stride, outp=16):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(3):
            layers.append(DenseLayer(inp + i * outp, outp, kernel_size, stride))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


# DenseFuse network
class DenseFuse(nn.Module):
    def __init__(self, num_channel):
        super(DenseFuse, self).__init__()

        denseblock = DenseBlock
        enc_channel = 16
        dec_channels = [64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(num_channel, enc_channel, kernel_size, stride)
        self.DB1 = denseblock(enc_channel, kernel_size, stride)

        # decoder
        self.conv2 = ConvLayer(dec_channels[0], dec_channels[0], kernel_size, stride)
        self.conv3 = ConvLayer(dec_channels[0], dec_channels[1], kernel_size, stride)
        self.conv4 = ConvLayer(dec_channels[1], dec_channels[2], kernel_size, stride)
        self.conv5 = ConvLayer(dec_channels[2], num_channel, kernel_size, stride, is_last=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

                if m.bias is not None:
                    m.bias.data.zero_()

    def encoder(self, x):
        enc = self.conv1(x)
        enc = self.DB1(enc)
        return enc

    def decoder(self, enc):
        dec = self.conv2(enc)
        dec = self.conv3(dec)
        dec = self.conv4(dec)
        dec = self.conv5(dec)
        return dec

    def fusion(self, enc1, enc2, strategy_type='addition'):
        fused = (enc1[0] + enc2[0]) / 2
        return fused
    # def fusion(self, en1, en2, strategy_type='addition'):
    #     # addition
    #     if strategy_type is 'attention_weight':
    #         # attention weight
    #         fusion_function = fusion_strategy.attention_fusion_weight
    #     else:
    #         fusion_function = fusion_strategy.addition_fusion
    #
    #     f_0 = fusion_function(en1[0], en2[0])
    #     return [f_0]
