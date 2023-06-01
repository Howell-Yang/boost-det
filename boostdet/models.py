import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from torchvision.ops import box_iou
import numpy as np
import cv2
from losses import GeneralizedFocalLoss

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=3, stride=1):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = repeat
        self.stride = stride
        self.in_project = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(repeat):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.out_project = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    def forward(self, x):
        x = self.in_project(x)
        for i in range(self.repeat):
            x_out = self.blocks[i](x)
            x = x_out + x
        x = self.out_project(x)
        return x

class FCOS(nn.Module):
    def __init__(self, num_classes):
        super(FCOS, self).__init__()
        self.num_classes = num_classes
        # Define your backbone network here
        repeat = 0
        self.backbone_stride4 = nn.Sequential(
            ResBlock(5, 32, repeat=repeat, stride=1), # 256x256
            ResBlock(32, 32, repeat=repeat, stride=2), # 128x128
            ResBlock(32, 32, repeat=repeat, stride=2), # 64x64,
        )
        self.backbone_stride16 = nn.Sequential(
            ResBlock(32, 32, repeat=repeat, stride=2), # 32x32,
            ResBlock(32, 32, repeat=repeat, stride=2), # 16x16,
        )
        in_channels = 32

        # 直接在输入部分加入embeddig? 也可以在输出部分加入embedding
        x_embedding = torch.arange(256, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)/256.0 # 1, 1, 1, 256
        y_embedding = torch.arange(256, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1)/256.0 # 1, 1, 256, 1
        x_embedding = x_embedding.expand(-1, -1, 256, -1) # 1, 256, 256, 1
        y_embedding = y_embedding.expand(-1, -1, -1, 256) # 1, 1, 256, 256
        xy_embedding = torch.cat([x_embedding, y_embedding], dim=1) # 1, 2, 256, 256
        self.position_embedding = nn.Parameter(xy_embedding, requires_grad=True)

        # Define the classification, regression, and centerness heads
        self.reg_max = 7
        self.cls_head = nn.Conv2d(in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.reg_head = nn.Conv2d(in_channels, self.num_classes * 4 * (2 * self.reg_max + 1), kernel_size=3, stride=1, padding=1)
        self.gfl_loss = GeneralizedFocalLoss(reg_max=self.reg_max)

    def forward(self, x):
        # Forward pass through the backbone network
        # Use the outputs to calculate class, regression, and centerness predictions
        # import numpy as np
        # print(np.shape(x), np.shape(self.position_embedding)) # torch.Size([8, 3, 256, 256]) torch.Size([1, 2, 256, 256])
        B = x.size(0)
        position_embedding = self.position_embedding.expand((B, -1, -1, -1))
        x = torch.cat([x, position_embedding], dim=1) 
        x4 = self.backbone_stride4(x)
        x16 = self.backbone_stride16(x4)

        x16_cls_logits = self.cls_head(x16)
        x16_reg_logits = self.reg_head(x16)
        return x16_cls_logits, x16_reg_logits

    def loss(self, x16_cls_logits, x16_reg_logits, targets):
        loss_qfl, loss_bbox, loss_dfl  = self.gfl_loss((x16_cls_logits, x16_reg_logits) , targets)
        loss = loss_qfl + loss_bbox + loss_dfl
        loss_dict = dict(
            loss_qfl = loss_qfl,
            loss_bbox = loss_bbox,
            loss_dfl = loss_dfl,
            loss = loss
        )
        return loss_dict

