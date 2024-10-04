# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math
from toolbox.paper3.paper3_10.largekernel import ALK
from toolbox.paper3.paper3_10.TR import TR



# Feature Integrity Learning and Refinement
eps = 1e-12


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                nn.AdaptiveAvgPool1d,
                nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2 * math.pi)
        self.initialize()

    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        a = a_in.view(b, l, -1, 1, 1)

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_ * (self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq * self.ln_2pi), dim=-1) \
                 - torch.sum((v - mu) ** 2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l ** (1 / 2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out

    def initialize(self):
        weight_init(self)


class IAE(nn.Module):
    def __init__(self):
        super(IAE, self).__init__()

        self.bn_trans = nn.BatchNorm2d(64)

        self.num_caps = 8

        planes = 16

        self.conv_m = nn.Conv2d(64, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(64, self.num_caps * 16, kernel_size=5, stride=1, padding=1, bias=False)

        self.bn_m = nn.BatchNorm2d(self.num_caps)
        self.bn_pose = nn.BatchNorm2d(self.num_caps * 16)

        self.emrouting = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3, stride=2, padding=1)
        self.bn_caps = nn.BatchNorm2d(self.num_caps * planes)

        self.conv_rec = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_rec = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)

        self.fuse1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))


        self.fuse2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.fuse3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.fuse4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.fuse5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.TR= TR(320)
        self.conv_fuse = nn.Sequential(nn.Conv2d(320, 320, kernel_size=1),
                                       nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))

        self.out1_ALK = ALK(64)
        self.out2_ALK = ALK(64)
        self.out3_ALK = ALK(64)
        self.out4_ALK = ALK(64)

    def forward(self, input1, input2, input3, input4, input5, fre):
        _, _, h1, w1 = input1.size()  # (8, 8)
        _, _, h2, w2 = input2.size()  # (16, 16)
        _, _, h3, w3 = input3.size()  # (32, 32)
        _, _, h4, w4 = input4.size()  # (64, 64)
        _, _, h5, w5 = input5.size()  # ()

        input1_up = F.interpolate(input1, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,32,32)
        input2_up = F.interpolate(input2, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,32,32)
        input4_up = F.interpolate(input4, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,32,32)
        input5_up = F.interpolate(input5, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,32,32)

        input_fuse = torch.cat((input1_up, input2_up, input3, input4_up, input5_up), dim=1)  # (8,64*5,64,64)
        input_fuse = self.conv_fuse(input_fuse + input_fuse * self.TR(input_fuse))

        # primary caps
        m, pose = self.conv_m(input_fuse), self.conv_pose(input_fuse)  # m:(b,8,64,64); pose:(b,8*16,64,64)
        m, pose = torch.sigmoid(self.bn_m(m)), self.bn_pose(pose)  # m:(b,8,64,64); pose:(b,8*16,64,64)

        m, pose = self.emrouting(m, pose)  # m:(b,8,32,32); pose:(b,128,32,32)
        pose = self.bn_caps(pose)  # pose:(b,8*16,32,32)

        pose = F.relu(self.bn_rec(self.conv_rec(pose)), inplace=True)  # pose:(b,64,32,32)

        pose1 = F.interpolate(pose, size=(h1, w1), mode='bilinear', align_corners=True)  # (8,64,8,8)
        pose2 = F.interpolate(pose, size=(h2, w2), mode='bilinear', align_corners=True)  # (8,64,16,16)
        pose3 = F.interpolate(pose, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,32,32)
        pose4 = F.interpolate(pose, size=(h4, w4), mode='bilinear', align_corners=True)  # (8,64,64,64)
        pose5 = F.interpolate(pose, size=(h5, w5), mode='bilinear', align_corners=True) #[4, 64, 128, 128])

        out1 = torch.cat((input1, pose1), dim=1)  # (8,128,8,8)
        out2 = torch.cat((input2, pose2), dim=1)  # (8,128,16,16)
        out3 = torch.cat((input3, pose3), dim=1)  # (8,128,32,32)
        out4 = torch.cat((input4, pose4), dim=1)  # (8,128,64,64)
        out5 = torch.cat((input5, pose5), dim=1)

        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)  # ([4, 64, 8, 8])
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)  # [4, 64, 16, 16])
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)  # ([4, 64, 32, 32])
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)  # [4, 64, 64, 64])
        out5 = F.relu(self.bn5(self.conv5(out5)), inplace=True)  #[4, 64, 128, 128])

        fre = F.interpolate(fre, scale_factor=0.25, mode='bilinear')
        out1 = self.fuse1(out1 * fre) + out1    # (8,64,8,8)
        out1 = F.interpolate(out1, size=(h2, w2), mode='bilinear', align_corners=True)  # (8,64,16,16)\\
        out1 = self.out1_ALK(out1)

        out2 = self.fuse2(out2 * out1) + out2  # (8,64,16,16)
        out2 = F.interpolate(out2, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,32,32)
        out2 = self.out2_ALK(out2)

        out3 = self.fuse3(out3 * out2) + out3  # (8,64,32,32)
        out3 = F.interpolate(out3, size=(h4, w4), mode='bilinear', align_corners=True)# (8,64,64,64)
        out3 = self.out3_ALK(out3)

        out4 = self.fuse4(out4 * out3) + out4
        out4 = F.interpolate(out4, size=(h5, w5), mode='bilinear', align_corners=True)
        out4 = self.out4_ALK(out4)

        out5 = self.fuse5(out5 * out4) + out5  # (8,64,64,64)

        # return out3
        return out5, out4, out3, out2, out1
