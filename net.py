import torch
import torch.nn as nn
from torch.nn import functional as F

from toolbox.backbone.ResNet import resnet50

from toolbox.paper3.paper3_11.SFR import SFR

from toolbox.paper3.paper3_11.capsulelarge3 import IAE

from toolbox.paper3.paper3_11.SAD import SAD

from toolbox.paper3.paper3_11.CGF import CGF

"""
    """
###############################################################################

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class net(nn.Module):
    def __init__(self,  channels=[64, 256, 512, 1024, 2048]):
        super(net, self).__init__()
        self.channels = channels

        resnet_raw_model1 = resnet50(pretrained=True)
        resnet_raw_model2 = resnet50(pretrained=True)
        ###############################################
        # Backbone model
        self.encoder_thermal_conv1 = resnet_raw_model1.conv1
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool

        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool

        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ###############################################
        # funsion encoders #


        self.fu_0 = SFR(self.channels[0])

        self.fu_1 = SFR(self.channels[1])

        self.fu_2 = SFR(self.channels[2])

        self.fu_3 = SFR(self.channels[3])

        self.fu_4 = SFR(self.channels[4])

        self.FRE2 = CGF(64,3)
        self.FRE3 = CGF(64,3)
        self.FRE4 = CGF(64,3)


        self.f_csm1_conv = conv3x3_bn_relu(256, 64)
        self.f_csm2_conv = conv3x3_bn_relu(512, 64)
        self.f_csm3_conv = conv3x3_bn_relu(1024, 64)
        self.f_csm4_conv = conv3x3_bn_relu(2048, 64)

        self.R_csm2_conv = conv3x3_bn_relu(512, 64)
        self.R_csm3_conv = conv3x3_bn_relu(1024, 64)
        self.R_csm4_conv = conv3x3_bn_relu(2048, 64)

        ###############################################
        # decoders #
        ###############################################
        # enhance receive field #
        self.SAD = SAD(64)
        self.FILR = IAE()

        self.coarse = nn.Sequential(
            nn.Conv2d(64, 6, 1, 1, 0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

        self.out0 = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.out1 = nn.Sequential(
            nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                                          )

        self.out2 = nn.Sequential(
            nn.Conv2d(64, 6, 1, 1, 0),
            nn.Upsample(scale_factor=4,mode='bilinear', align_corners=True)
        )

        self.out3 = nn.Sequential(
            nn.Conv2d(64, 6, 1, 1, 0),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )

        self.out4 = nn.Sequential(
            nn.Conv2d(64, 6, 1, 1, 0),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        )



                

    def forward(self, rgb, d):
        ###############################################
        # Backbone model
        rgb0 = self.encoder_rgb_conv1(rgb)
        rgb0 = self.encoder_rgb_bn1(rgb0)
        rgb0 = self.encoder_rgb_relu(rgb0)

        d0 = self.encoder_thermal_conv1(d)
        d0 = self.encoder_thermal_bn1(d0)
        d0 = self.encoder_thermal_relu(d0)
        ####################################################
        ## fusion
        ####################################################

        f0 = self.fu_0(rgb0, d0)
        F0 = f0

        rgb1 = self.encoder_rgb_maxpool(rgb0)
        rgb1 = self.encoder_rgb_layer1(rgb1)

        d1 = self.encoder_thermal_maxpool(d0)
        d1 = self.encoder_thermal_layer1(d1)

        ## layer1 融合

        f1 = self.fu_1(rgb1, d1)
        F1 = self.f_csm1_conv(f1)
        ## 传输到encoder2


        rgb2 = self.encoder_rgb_layer2(rgb1)
        d2 = self.encoder_thermal_layer2(d1)

        ## layer2 融合
        f2 = self.fu_2(rgb2, d2)
        F2 = self.f_csm2_conv(f2)
        RGB2 = self.R_csm2_conv(rgb2)
        ## 传输到encoder3

        rgb3 = self.encoder_rgb_layer3(rgb2)
        d3 = self.encoder_thermal_layer3(d2)

        ## layer3 融合
        f3 = self.fu_3(rgb3, d3)
        F3 = self.f_csm3_conv(f3)
        RGB3 = self.R_csm3_conv(rgb3)
        ## 传输到encoder4


        rgb4 = self.encoder_rgb_layer4(rgb3)
        d4 = self.encoder_thermal_layer4(d3)

        ## layer4 融合
        f4 = self.fu_4(rgb4, d4)
        F4 = self.f_csm4_conv(f4)
        RGB4 = self.R_csm4_conv(rgb4)

        ####################################################
        ## fre
        ####################################################
        RGB4 = self.FRE4(RGB4)
        RGB3 = self.FRE3(RGB3)
        RGB2 = self.FRE2(RGB2)
        fre = self.SAD(RGB4, RGB3, RGB2)
        ####################################################
        ## decoder
        ####################################################
        ## enhance

        coarse = self.coarse(fre)

        out0, out1, out2, out3, out4 = self.FILR(F4, F3, F2, F1, F0, fre)

        #
        out0 = self.out0(out0)
        out1 = self.out1(out1)
        out2 = self.out2(out2)
        out3 = self.out3(out3)
        out4 = self.out4(out4)

        return out0, out1, out2, out3, out4, coarse
        # return f1, f2, f3, f4



if __name__ == '__main__':
    rgb = torch.randn(4, 3, 256, 256)
    d = torch.randn(4, 3, 256, 256)

    model = net()

    a = model(rgb, d)

    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)
    print(a[3].shape)
    print(a[4].shape)



