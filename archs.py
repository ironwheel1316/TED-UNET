import torch
from torch import nn
import math
import torch.nn.functional as F

#模型架构

__all__ = ['UNet', 'NestedUNet', 'UNetPlusMinus','UNetPlusMinus2','WNet','WNetPlusPlus', 'UNetDeep', 'Conv3UNet','ConvUNet11223', 'ConvUNet32211', 'ConvUNet31122', 'ConvUNet32222', 'ConvUNet32221', 'ConvUNet3333', 'ConvUNet444', 'ConvUNet55','ConvUNet6',"ResUNet","UNetR1","UNetR12", "R1UNet444","R1UNet444UP","R1UNet444UP333","NOR1UNet444","UNet3"]


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class VGGBlock3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
        
class UNetPlusMinus(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        # self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        # self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        # self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            # self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0,  x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            # output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            # return [output1, output2, output3, output4]
            return [output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
        

class UNetPlusMinus2(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        # self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        # self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        # self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_3 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        # self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        # self.conv0_4 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_4 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            # self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            # self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0,  x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0,  x0_2, self.up(x1_3)], 1))

        if self.deep_supervision:
            # output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            # output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            # return [output1, output2, output3, output4]
            # return [output2, output3, output4]
            return [output2, output4]

        else:
            output = self.final(x0_4)
            return output
        

class WNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # --- UNet A: Encoder ---
        self.conv0_0_A = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0_A = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0_A = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0_A = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0_A = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4]) # Bottleneck A

        # --- UNet A: Decoder (partial, up to level 1 resolution) ---
        self.conv3_1_A = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2_A = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3_A = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1]) # Output of UNet A's decoder at level 1 ("W center")

        # --- UNet B: Encoder (starts from output of UNet A's partial decoder) ---
        # Input to conv1_0_B is x1_3_A, which has nb_filter[1] channels
        self.conv1_0_B = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_0_B = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2]) # Input is pool(output of conv1_0_B)
        self.conv3_0_B = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0_B = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4]) # Bottleneck B

        # --- UNet B: Decoder (full, up to original resolution) ---
        self.conv3_1_B = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2_B = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3_B = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        # Skip connection for conv0_4_B comes from conv0_0_A
        self.conv0_4_B = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        # --- Final Output Layer ---
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # --- UNet A: Encoder ---
        x0_0_A = self.conv0_0_A(input)
        p1_A = self.pool(x0_0_A)
        x1_0_A = self.conv1_0_A(p1_A)
        p2_A = self.pool(x1_0_A)
        x2_0_A = self.conv2_0_A(p2_A)
        p3_A = self.pool(x2_0_A)
        x3_0_A = self.conv3_0_A(p3_A)
        p4_A = self.pool(x3_0_A)
        x4_0_A = self.conv4_0_A(p4_A) # Bottleneck A

        # --- UNet A: Decoder (partial) ---
        u3_A = self.up(x4_0_A)
        x3_1_A = self.conv3_1_A(torch.cat([x3_0_A, u3_A], 1))
        u2_A = self.up(x3_1_A)
        x2_2_A = self.conv2_2_A(torch.cat([x2_0_A, u2_A], 1))
        u1_A = self.up(x2_2_A)
        x1_3_A = self.conv1_3_A(torch.cat([x1_0_A, u1_A], 1)) # "W center point" - input to UNet B

        # --- UNet B: Encoder ---
        # x1_3_A is at resolution level 1.
        x1_0_B = self.conv1_0_B(x1_3_A) # Processed features at level 1
        p2_B = self.pool(x1_0_B)
        x2_0_B = self.conv2_0_B(p2_B)    # Features at level 2
        p3_B = self.pool(x2_0_B)
        x3_0_B = self.conv3_0_B(p3_B)    # Features at level 3
        p4_B = self.pool(x3_0_B)
        x4_0_B = self.conv4_0_B(p4_B)    # Bottleneck B (at level 4)

        # --- UNet B: Decoder ---
        u3_B = self.up(x4_0_B)
        x3_1_B = self.conv3_1_B(torch.cat([x3_0_B, u3_B], 1))
        u2_B = self.up(x3_1_B)
        x2_2_B = self.conv2_2_B(torch.cat([x2_0_B, u2_B], 1))
        u1_B = self.up(x2_2_B)
        x1_3_B = self.conv1_3_B(torch.cat([x1_0_B, u1_B], 1))
        u0_B = self.up(x1_3_B)
        # Skip connection from the very first layer of UNet A (x0_0_A)
        x0_4_B = self.conv0_4_B(torch.cat([x0_0_A, u0_B], 1))

        output = self.final(x0_4_B)
        return output

# ... (VGGBlock, UNet, NestedUNet, UNetPlusMinus, UNetPlusMinus2, WNet) ...

class Conv3UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock3(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock3(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock3(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock3(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock3(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock3(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock3(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock3(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock3(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class UNetDeep(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5])

        # Decoder
        self.conv4_1 = VGGBlock(nb_filter[5]+nb_filter[4], nb_filter[4], nb_filter[4])
        self.conv3_1 = VGGBlock(nb_filter[4]+nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_1 = VGGBlock(nb_filter[2]+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv0_1 = VGGBlock(nb_filter[1]+nb_filter[0], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x5_0 = self.conv5_0(self.pool(x4_0))

        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_1)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))

        output = self.final(x0_1)
        return output

class UNet_A_Core(nn.Module): # Based on NestedUNet
    def __init__(self, input_channels=3, deep_supervision_enabled=False, **kwargs): # No num_classes needed here
        super().__init__()
        # We use deep_supervision_enabled to control if final layers are built,
        # but for WNetPlusPlus, UNet_A_Core won't use its own final layers for the WNet output.
        # It's mainly to keep the NestedUNet structure consistent if you were to use it standalone.

        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision_enabled = deep_supervision_enabled # Store for potential internal use

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1]) # This will be W_center

        # self.conv0_4 and final layers are not strictly needed for WNetPlusPlus's UNet_A part
        # as we only need x0_0 and x1_3 from it.
        # However, to keep it a "runnable" NestedUNet if used alone, we can include them.
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        if self.deep_supervision_enabled: # Only build final layers if deep supervision is on for this part
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1) # Dummy num_classes=1
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1) # Dummy num_classes=1


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)) # We don't need x0_3 for WNetPlusPlus

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1)) # This is W_center

        # For WNetPlusPlus, we only need x0_0 and x1_3
        return x0_0, x1_3
    
# ... (UNet_A_Core defined above) ...

class UNet_B_Core(nn.Module): # Based on NestedUNet
    def __init__(self, num_classes, channels_x0_0_A, channels_W_center, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512] # Standard UNet++ filters
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder for B starts from W_center (which is at level 1 resolution)
        # So, conv1_0_B takes W_center as input
        self.conv1_0_B = VGGBlock(channels_W_center, nb_filter[1], nb_filter[1])
        self.conv2_0_B = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0_B = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0_B = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4]) # Bottleneck of B

        # Decoder for B - dense connections
        # x0_1_B: combines x0_0_A (skip) and up(x1_0_B)
        self.conv0_1_B = VGGBlock(channels_x0_0_A + nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv1_1_B = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1]) # From x1_0_B and up(x2_0_B)
        self.conv2_1_B = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1_B = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        # x0_2_B: combines x0_0_A, x0_1_B, and up(x1_1_B)
        self.conv0_2_B = VGGBlock(channels_x0_0_A + nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2_B = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2_B = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        # x0_3_B: combines x0_0_A, x0_1_B, x0_2_B, and up(x1_2_B)
        self.conv0_3_B = VGGBlock(channels_x0_0_A + nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3_B = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        # x0_4_B: combines x0_0_A, x0_1_B, x0_2_B, x0_3_B, and up(x1_3_B)
        self.conv0_4_B = VGGBlock(channels_x0_0_A + nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0])


        if self.deep_supervision:
            self.final1_B = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2_B = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3_B = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4_B = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final_B = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x0_0_A_skip, W_center_input):
        # --- UNet B: Encoder ---
        # W_center_input is x1_3_A from UNet_A_Core
        x1_0_B = self.conv1_0_B(W_center_input)
        x2_0_B = self.conv2_0_B(self.pool(x1_0_B))
        x3_0_B = self.conv3_0_B(self.pool(x2_0_B))
        x4_0_B = self.conv4_0_B(self.pool(x3_0_B)) # Bottleneck B

        # --- UNet B: Decoder (Nested) ---
        # Level 0 skips (x0_j_B) will use x0_0_A_skip
        x0_1_B = self.conv0_1_B(torch.cat([x0_0_A_skip, self.up(x1_0_B)], 1))

        x1_1_B = self.conv1_1_B(torch.cat([x1_0_B, self.up(x2_0_B)], 1))
        x0_2_B = self.conv0_2_B(torch.cat([x0_0_A_skip, x0_1_B, self.up(x1_1_B)], 1))

        x2_1_B = self.conv2_1_B(torch.cat([x2_0_B, self.up(x3_0_B)], 1))
        x1_2_B = self.conv1_2_B(torch.cat([x1_0_B, x1_1_B, self.up(x2_1_B)], 1))
        x0_3_B = self.conv0_3_B(torch.cat([x0_0_A_skip, x0_1_B, x0_2_B, self.up(x1_2_B)], 1))

        x3_1_B = self.conv3_1_B(torch.cat([x3_0_B, self.up(x4_0_B)], 1))
        x2_2_B = self.conv2_2_B(torch.cat([x2_0_B, x2_1_B, self.up(x3_1_B)], 1))
        x1_3_B = self.conv1_3_B(torch.cat([x1_0_B, x1_1_B, x1_2_B, self.up(x2_2_B)], 1))
        x0_4_B = self.conv0_4_B(torch.cat([x0_0_A_skip, x0_1_B, x0_2_B, x0_3_B, self.up(x1_3_B)], 1))


        if self.deep_supervision:
            output1 = self.final1_B(x0_1_B)
            output2 = self.final2_B(x0_2_B)
            output3 = self.final3_B(x0_3_B)
            output4 = self.final4_B(x0_4_B)
            return [output1, output2, output3, output4]
        else:
            output = self.final_B(x0_4_B)
            return output
        

# ... (UNet_A_Core and UNet_B_Core defined above) ...

class WNetPlusPlus(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision_B=False, **kwargs):
        super().__init__()

        # For UNet_A_Core, we don't need its own deep supervision for the final WNetPlusPlus output.
        # It's just an internal feature extractor.
        self.unet_A_core = UNet_A_Core(input_channels=input_channels)

        # Determine channel counts from unet_A_core's structure
        # Based on nb_filter = [32, 64, 128, 256, 512]
        # x0_0_A will have nb_filter[0] channels (32)
        # x1_3_A (W_center) will have nb_filter[1] channels (64)
        channels_x0_0_A = 32
        channels_W_center = 64

        self.unet_B_core = UNet_B_Core(num_classes=num_classes,
                                       channels_x0_0_A=channels_x0_0_A,
                                       channels_W_center=channels_W_center,
                                       deep_supervision=deep_supervision_B)

    def forward(self, input):
        x0_0_A, w_center = self.unet_A_core(input)
        output = self.unet_B_core(x0_0_A, w_center)
        return output


class VGGBlock1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ConvUNet11223(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder with custom convolution counts
        self.conv0_0 = VGGBlock1(input_channels, nb_filter[0], nb_filter[0])  # Layer 0: 1 conv
        self.conv1_0 = VGGBlock1(nb_filter[0], nb_filter[1], nb_filter[1])    # Layer 1: 1 conv
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])     # Layer 2: 2 convs
        self.conv3_0 = VGGBlock3(nb_filter[2], nb_filter[3], nb_filter[3])    # Layer 3: 3 convs
        self.conv4_0 = VGGBlock3(nb_filter[3], nb_filter[4], nb_filter[4])    # Layer 4: 3 convs

        # Decoder with custom convolution counts
        self.conv3_1 = VGGBlock3(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])  # Layer 3: 3 convs
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])   # Layer 2: 2 convs
        self.conv1_3 = VGGBlock1(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])  # Layer 1: 1 conv
        self.conv0_4 = VGGBlock1(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])  # Layer 0: 1 conv

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class ConvUNet32211(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder with custom convolution counts
        self.conv0_0 = VGGBlock3(input_channels, nb_filter[0], nb_filter[0])  # Layer 0: 3 convs
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])     # Layer 1: 2 convs
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])     # Layer 2: 2 convs
        self.conv3_0 = VGGBlock1(nb_filter[2], nb_filter[3], nb_filter[3])    # Layer 3: 1 conv
        self.conv4_0 = VGGBlock1(nb_filter[3], nb_filter[4], nb_filter[4])    # Layer 4: 1 conv

        # Decoder with custom convolution counts
        self.conv3_1 = VGGBlock1(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])  # Layer 3: 1 conv
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])   # Layer 2: 2 convs
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])   # Layer 1: 2 convs
        self.conv0_4 = VGGBlock3(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])  # Layer 0: 3 convs

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class ConvUNet31122(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder with custom convolution counts
        self.conv0_0 = VGGBlock3(input_channels, nb_filter[0], nb_filter[0])  # Layer 0: 3 convs
        self.conv1_0 = VGGBlock1(nb_filter[0], nb_filter[1], nb_filter[1])    # Layer 1: 1 conv
        self.conv2_0 = VGGBlock1(nb_filter[1], nb_filter[2], nb_filter[2])    # Layer 2: 1 conv
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])     # Layer 3: 2 convs
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])     # Layer 4: 2 convs

        # Decoder with custom convolution counts
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])   # Layer 3: 2 convs
        self.conv2_2 = VGGBlock1(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])  # Layer 2: 1 conv
        self.conv1_3 = VGGBlock1(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])  # Layer 1: 1 conv
        self.conv0_4 = VGGBlock3(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])  # Layer 0: 3 convs

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class ConvUNet32222(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder with custom convolution counts
        self.conv0_0 = VGGBlock3(input_channels, nb_filter[0], nb_filter[0])  # Layer 0: 3 convs
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])     # Layer 1: 2 convs
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])     # Layer 2: 2 convs
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])     # Layer 3: 2 convs
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])     # Layer 4: 2 convs

        # Decoder with custom convolution counts
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])   # Layer 3: 2 convs
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])   # Layer 2: 2 convs
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])   # Layer 1: 2 convs
        self.conv0_4 = VGGBlock3(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])  # Layer 0: 3 convs

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class ConvUNet32221(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder with custom convolution counts
        self.conv0_0 = VGGBlock3(input_channels, nb_filter[0], nb_filter[0])  # Layer 0: 3 convs
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])     # Layer 1: 2 convs
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])     # Layer 2: 2 convs
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])     # Layer 3: 2 convs
        self.conv4_0 = VGGBlock1(nb_filter[3], nb_filter[4], nb_filter[4])    # Layer 4: 1 conv

        # Decoder with custom convolution counts
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])   # Layer 3: 2 convs
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])   # Layer 2: 2 convs
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])   # Layer 1: 2 convs
        self.conv0_4 = VGGBlock3(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])  # Layer 0: 3 convs

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class ConvUNet3333(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock3(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock3(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock3(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock3(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv2_1 = VGGBlock3(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock3(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock3(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))

        output = self.final(x0_3)
        return output


class VGGBlock4(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(middle_channels)
        self.conv4 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        return out


class ConvUNet444(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock4(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock4(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock4(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock4(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv2_1 = VGGBlock4(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock4(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock4(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0],            num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))

        output = self.final(x0_3)
        return output


class VGGBlock5(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(middle_channels)
        self.conv4 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(middle_channels)
        self.conv5 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        return out

class VGGBlock6(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(middle_channels)
        self.conv4 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(middle_channels)
        self.conv5 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(middle_channels)
        self.conv6 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)

        return out
    


class ConvUNet6(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock6(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock6(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv0_1 = VGGBlock6(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        output = self.final(x0_1)
        return output

class ConvUNet55(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock5(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock5(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock5(nb_filter[1], nb_filter[2], nb_filter[2])

        self.conv1_1 = VGGBlock5(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_2 = VGGBlock5(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up(x1_1)], 1))

        output = self.final(x0_2)
        return output
    
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers_list = []
        layers_list.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers_list.append(block(self.inplanes, planes))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        # Encoder path, returning features for skip connections
        s0 = self.conv1(x)
        s0 = self.bn1(s0)
        s0_relu = self.relu(s0) # Skip 0: 64 channels, H/2, W/2 (after initial conv, bn, relu)
        
        s1_in = self.maxpool(s0_relu)
        s1 = self.layer1(s1_in)      # Skip 1: 256 channels (64 * 4), H/4, W/4
        s2 = self.layer2(s1)         # Skip 2: 512 channels (128 * 4), H/8, W/8
        s3 = self.layer3(s2)         # Skip 3: 1024 channels (256 * 4), H/16, W/16
        
        s4_bottleneck = self.layer4(s3) # Bottleneck: 2048 channels (512 * 4), H/32, W/32

        return [s0_relu, s1, s2, s3, s4_bottleneck]

# --- End of ResNet definition ---


class ResUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, resnet_block_type=Bottleneck, resnet_layers_config=[3,4,6,3], **kwargs):
        super().__init__()

        self.encoder = ResNet(block=resnet_block_type, layers=resnet_layers_config, input_channels=input_channels)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Output channels of ResNet encoder stages (Bottleneck expansion = 4):
        # s0_relu: 64
        # s1 (layer1 out): 64 * 4 = 256
        # s2 (layer2 out): 128 * 4 = 512
        # s3 (layer3 out): 256 * 4 = 1024
        # s4_bottleneck (layer4 out): 512 * 4 = 2048

        # Decoder VGGBlock output channels
        # These are chosen to gradually reduce channels, similar to standard U-Net
        # The VGGBlock(in, middle, out) will have middle=out for these.
        decoder_block_out_channels = {
            "center": 512, # After processing the bottleneck
            "d3": 256,     # Decoder for s3 skip
            "d2": 128,     # Decoder for s2 skip
            "d1": 64,      # Decoder for s1 skip
            "d0": 32       # Decoder for s0 skip
        }

        # Center block (after encoder bottleneck)
        self.center_conv = VGGBlock(2048, decoder_block_out_channels["center"], decoder_block_out_channels["center"])

        # Decoder blocks
        self.dec3 = VGGBlock(decoder_block_out_channels["center"] + 1024, decoder_block_out_channels["d3"], decoder_block_out_channels["d3"])
        self.dec2 = VGGBlock(decoder_block_out_channels["d3"] + 512,  decoder_block_out_channels["d2"], decoder_block_out_channels["d2"])
        self.dec1 = VGGBlock(decoder_block_out_channels["d2"] + 256,  decoder_block_out_channels["d1"], decoder_block_out_channels["d1"])
        self.dec0 = VGGBlock(decoder_block_out_channels["d1"] + 64,   decoder_block_out_channels["d0"], decoder_block_out_channels["d0"])
        
        # Final upsampling (if needed, ResNet's first conv already reduces by 2, so dec0 is at H/2)
        # This upsample brings H/2 to H
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(decoder_block_out_channels["d0"], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = self.encoder(x)
        s0_relu, s1, s2, s3, s4_bottleneck = skips[0], skips[1], skips[2], skips[3], skips[4]

        # Decoder
        center = self.center_conv(s4_bottleneck) # (H/32, W/32)

        up3 = self.up(center) # (H/16, W/16)
        cat3 = torch.cat([up3, s3], 1)
        d3 = self.dec3(cat3)

        up2 = self.up(d3) # (H/8, W/8)
        cat2 = torch.cat([up2, s2], 1)
        d2 = self.dec2(cat2)

        up1 = self.up(d2) # (H/4, W/4)
        cat1 = torch.cat([up1, s1], 1)
        d1 = self.dec1(cat1)

        up0 = self.up(d1) # (H/2, W/2)
        cat0 = torch.cat([up0, s0_relu], 1)
        d0 = self.dec0(cat0)

        # Final output
        out = self.final_up(d0) # (H, W)
        out = self.final_conv(out)

        return out


class UNetR1(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 修正变量名：将 conv1*1_0 改为 conv1x1_0 (或其他有效名称)
        self.conv1x1_0 = nn.Conv2d(input_channels, nb_filter[0], kernel_size=1, padding=0)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input_tensor): # Renamed 'input' to 'input_tensor' to avoid conflict with built-in
        x0_0 = self.conv0_0(input_tensor)
        # 使用修正后的变量名
        x0_0_0 = self.conv1x1_0(input_tensor)
        x1_0 = self.conv1_0(self.pool(x0_0))
# ...existing code...
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        # x0_0 + x0_0_0 是两个并行路径输出的特征图相加，然后结果与其他特征拼接
        # 这是一种特征融合方式，而不是典型的将块的输入加到块的输出上的残差连接。
        fused_x0_0 = x0_0 + x0_0_0 
        x0_4 = self.conv0_4(torch.cat([fused_x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    

class UNetR12(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 修正变量名：将 conv1*1_0 改为 conv1x1_0 (或其他有效名称)
        self.conv1x1_0 = nn.Conv2d(input_channels, nb_filter[0], kernel_size=1, padding=0)
        self.conv1x1_1 = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=1, padding=0)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input_tensor): # Renamed 'input' to 'input_tensor' to avoid conflict with built-in
        x0_0 = self.conv0_0(input_tensor)
        # 使用修正后的变量名
        x0_0_0 = self.conv1x1_0(input_tensor)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0_0 = self.conv1x1_1(self.pool(x0_0))
# ...existing code...
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        # x0_0 + x0_0_0 是两个并行路径输出的特征图相加，然后结果与其他特征拼接
        # 这是一种特征融合方式，而不是典型的将块的输入加到块的输出上的残差连接。
        fused_x0_0 = x0_0 + x0_0_0 
        fused_x1_0 = x1_0 + x1_0_0
        x1_3 = self.conv1_3(torch.cat([fused_x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([fused_x0_0, self.up(x1_3)], 1))


        output = self.final(x0_4)
        return output
    

class R1UNet444(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1_0 = nn.Conv2d(input_channels, nb_filter[0], kernel_size=1, padding=0)

        self.conv0_0 = VGGBlock4(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock4(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock4(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock4(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv2_1 = VGGBlock4(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock4(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock4(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0],num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0_0 = self.conv1x1_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        fused_x0_0 = x0_0 + x0_0_0 
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([fused_x0_0, self.up(x1_2)], 1))

        output = self.final(x0_3)
        return output
    

class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(EUCB, self).__init__()
        self.scale_factor = scale_factor
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)  # 深度可分离卷积
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)  # 1x1卷积
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)  # 上采样
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        return x

class R1UNet444UP(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.EUCB3 = EUCB(nb_filter[3], nb_filter[3], scale_factor=2)  # 用于上采样 x3_0
        self.conv1x1_0 = nn.Conv2d(input_channels, nb_filter[0], kernel_size=1, padding=0)

        self.conv0_0 = VGGBlock4(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock4(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock4(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock4(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv2_1 = VGGBlock4(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock4(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock4(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0_0 = self.conv1x1_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        fused_x0_0 = x0_0 + x0_0_0
        x2_1 = self.conv2_1(torch.cat([x2_0, self.EUCB3(x3_0)], 1))  # 使用 EUCB3 上采样 x3_0
        x1_2 = self.conv1_2(torch.cat([self.up(x2_1),x1_0], 1))  # 使用 EUCB2 上采样 x2_1
        x0_3 = self.conv0_3(torch.cat([fused_x0_0,self.up(x1_2)], 1))  # 使用 EUCB1 上采样 x1_2

        output = self.final(x0_3)
        return output
    

class R1UNet444UP333(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.EUCB3 = EUCB(nb_filter[3], nb_filter[3], scale_factor=2)  # 用于上采样 x3_0
        self.EUCB2 = EUCB(nb_filter[2], nb_filter[2], scale_factor=2)  # 用于上采样 x2_1
        self.EUCB1 = EUCB(nb_filter[1], nb_filter[1], scale_factor=2)  # 用于上采样 x1_2
        self.conv1x1_0 = nn.Conv2d(input_channels, nb_filter[0], kernel_size=1, padding=0)

        self.conv0_0 = VGGBlock4(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock4(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock4(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock4(nb_filter[2], nb_filter[3], nb_filter[3])

        self.conv2_1 = VGGBlock3(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock3(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock3(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0_0 = self.conv1x1_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        fused_x0_0 = x0_0 + x0_0_0
        x2_1 = self.conv2_1(torch.cat([x2_0, self.EUCB3(x3_0)], 1))  # 使用 EUCB3 上采样 x3_0
        x1_2 = self.conv1_2(torch.cat([x1_0, self.EUCB2(x2_1)], 1))  # 使用 EUCB2 上采样 x2_1
        x0_3 = self.conv0_3(torch.cat([fused_x0_0, self.EUCB1(x1_2)], 1))  # 使用 EUCB1 上采样 x1_2

        output = self.final(x0_3)
        return output    

    
class UNet3(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])

        # Decoder
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        # Encoder
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        # Decoder
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))

        output = self.final(x0_3)
        return output