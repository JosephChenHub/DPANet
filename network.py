#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50

from lib.utils import load_model






class SA(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)



""" fusion two level features """
class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, norm_layer=nn.BatchNorm2d):
        super(FAM, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256*2, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(256)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)




class CrossAttention(nn.Module):
    def __init__(self, in_channel=256, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_key   = nn.Conv2d(in_channel, in_channel//ratio, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, rgb, depth):
        bz, c, h, w = rgb.shape
        depth_q = self.conv_query(depth).view(bz, -1, h*w).permute(0, 2, 1)
        depth_k = self.conv_key(depth).view(bz, -1, h*w)
        mask  = torch.bmm(depth_q, depth_k) #bz, hw, hw
        mask  = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_value(rgb).view(bz, c, -1)
        feat  = torch.bmm(rgb_v, mask.permute(0,2,1)) # bz, c, hw
        feat  = feat.view(bz, c, h, w)

        return feat


class CMAT(nn.Module):
    def __init__(self, in_channel, CA=True, ratio=8):
        super(CMAT, self).__init__()
        self.CA = CA

        self.sa1 = SA(in_channel)
        self.sa2 = SA(in_channel)
        if self.CA:
            self.att1 = CrossAttention(256, ratio=ratio)
            self.att2 = CrossAttention(256, ratio=ratio)
        else:
            print("Warning: not use CrossAttention!")
            self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
            self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)



    def forward(self, rgb, depth, beta, gamma, gate):
        rgb = self.sa1(rgb)
        depth = self.sa2(depth)
        if self.CA:
            feat_1 = self.att1(rgb, depth)
            feat_2 = self.att2(depth, rgb)
        else:
            w1 = self.conv2(rgb)
            w2 = self.conv3(depth)
            feat_1 = F.relu(w2*rgb, inplace=True)
            feat_2 = F.relu(w1*depth, inplace=True)

        out1 = rgb + gate *  beta * feat_1
        out2 = depth + (1.0-gate) * gamma * feat_2

        return out1, out2


class Fusion(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x1, x2, alpha, beta):
        out1 = alpha * x1 + beta*(1.0 - alpha) * x2
        out2 = x1 * x2
        out  = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv0(out)), inplace=True)

        return out

class Segment(nn.Module):
    def __init__(self, backbone='resnet18', norm_layer=nn.BatchNorm2d, cfg=None, aux_layers=True):
        super(Segment, self).__init__()
        self.cfg     = cfg
        self.aux_layers = aux_layers

        if backbone == 'resnet18':
            channels = [64, 128, 256, 512]
            self.backbone_rgb =  resnet18(in_channel=3, norm_layer=norm_layer)
            self.backbone_d = resnet18(in_channel=1, norm_layer=norm_layer)
            backbone_rgb = load_model(self.backbone_rgb, 'model_zoo/resnet18-5c106cde.pth')
            backbone_d = load_model(self.backbone_d, 'model_zoo/resnet18-5c106cde.pth', depth_input=True)
        elif backbone == 'resnet34':
            channels = [64, 128, 256, 512] # resnet34
            self.backbone_rgb =  resnet34(in_channel=3, norm_layer=norm_layer)
            self.backbone_d = resnet34(in_channel=1, norm_layer=norm_layer)
            backbone_rgb = load_model(self.backbone_rgb, 'model_zoo/resnet34-333f7ec4.pth')
            backbone_d = load_model(self.backbone_rgb, 'model_zoo/resnet34-333f7ec4.pth', depth_input=True)
        elif backbone == 'resnet50':
            channels = [256, 512, 1024, 2048]
            self.backbone_rgb =  resnet50(in_channel=3, norm_layer=norm_layer)
            self.backbone_d = resnet50(in_channel=1, norm_layer=norm_layer)
            backbone_rgb = load_model(self.backbone_rgb, 'model_zoo/resnet50-19c8e357.pth')
            backbone_d = load_model(self.backbone_rgb, 'model_zoo/resnet50-19c8e357.pth', depth_input=True)
        else:
            raise Exception("backbone:%s does not support!"%backbone)
        if backbone_rgb is None:
            print("Warning: the model_zoo of {} does no exist!".format(backbone))
        else:
            self.backbone_rgb = backbone_rgb
            self.backbone_d = backbone_d


        # fusion modules
        self.cmat5 = CMAT(channels[3], True, ratio=8)
        self.cmat4 = CMAT(channels[2], True, ratio=8)
        self.cmat3 = CMAT(channels[1], True, ratio=8)
        self.cmat2 = CMAT(channels[0], True, ratio=8)

        # low-level & high-level
        self.fam54_1 = FAM(256, 256)
        self.fam43_1 = FAM(256, 256)
        self.fam32_1 = FAM(256, 256)
        self.fam54_2 = FAM(256, 256)
        self.fam43_2 = FAM(256, 256)
        self.fam32_2 = FAM(256, 256)

        # fusion, TBD
        self.fusion = Fusion(256)

        if self.aux_layers:
            self.linear5_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear4_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear3_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear2_1 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear5_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear4_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear3_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
            self.linear2_2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.linear_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                   nn.Linear(channels[-1]*2, 512),
                   ##nn.Dropout(p=0.3),
                   nn.ReLU(True),
                   nn.Linear(512, 256+1),
                   nn.Sigmoid(),
                   )

        self.initialize()

    def forward(self, rgb, depth):
        raw_size = rgb.size()[2:]
        bz = rgb.shape[0]
        enc2_1, enc3_1, enc4_1, enc5_1 = self.backbone_rgb(rgb)
        enc2_2, enc3_2, enc4_2, enc5_2 = self.backbone_d(depth)

        rgb_gap = self.gap1(enc5_1)
        rgb_gap = rgb_gap.view(bz, -1)
        depth_gap = self.gap2(enc5_2)
        depth_gap = depth_gap.view(bz, -1)
        feat = torch.cat((rgb_gap, depth_gap), dim=1)
        feat = self.fc(feat)

        gate = feat[:, -1].view(bz, 1, 1, 1)

        alpha = feat[:, :256]
        alpha = alpha.view(bz, 256, 1, 1)


        out5_1, out5_2 = self.cmat5(enc5_1, enc5_2, 1, 1, gate)
        de4_1, de4_2   = self.cmat4(enc4_1, enc4_2, 1, 1, gate)
        de3_1, de3_2   = self.cmat3(enc3_1, enc3_2, 1, 1, gate)
        de2_1, de2_2   = self.cmat2(enc2_1, enc2_2, 1, 1, gate)

        out4_1 = self.fam54_1(de4_1, out5_1)
        out3_1 = self.fam43_1(de3_1, out4_1)
        out2_1 = self.fam32_1(de2_1, out3_1)

        out4_2 = self.fam54_2(de4_2, out5_2)
        out3_2 = self.fam43_2(de3_2, out4_2)
        out2_2 = self.fam32_2(de2_2, out3_2)


        # final fusion
        out = self.fusion(out2_1, out2_2, alpha, gate)
        out = F.interpolate(self.linear_out(out), size=raw_size, mode='bilinear', )
        # aux_layer
        if self.training and self.aux_layers:
            out5_1 = F.interpolate(self.linear5_1(out5_1), size=raw_size, mode='bilinear')
            out4_1 = F.interpolate(self.linear4_1(out4_1), size=raw_size, mode='bilinear')
            out3_1 = F.interpolate(self.linear3_1(out3_1), size=raw_size, mode='bilinear')
            out2_1 = F.interpolate(self.linear2_1(out2_1), size=raw_size, mode='bilinear')
            out5_2 = F.interpolate(self.linear5_2(out5_2), size=raw_size, mode='bilinear')
            out4_2 = F.interpolate(self.linear4_2(out4_2), size=raw_size, mode='bilinear')
            out3_2 = F.interpolate(self.linear3_2(out3_2), size=raw_size, mode='bilinear')
            out2_2 = F.interpolate(self.linear2_2(out2_2), size=raw_size, mode='bilinear')

            return out, out2_1, out3_1, out4_1, out5_1, out2_2, out3_2, out4_2, out5_2, gate.view(bz, -1)
        else:

            return [out, gate.view(bz, -1)]

    def initialize(self):
        if self.cfg and self.cfg.snapshot:
            print("loading state dict:%s ..."%(self.cfg.snapshot))
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            pass

