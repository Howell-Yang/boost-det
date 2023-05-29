import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from torchvision.ops import box_iou
import numpy as np
import cv2

class FCOS(nn.Module):
    def __init__(self, num_classes):
        super(FCOS, self).__init__()
        self.num_classes = num_classes
        # Define your backbone network here
        self.backbone_stride4 = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1), # 256x256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.backbone_stride16 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 直接在输入部分加入embeddig? 也可以在输出部分加入embedding
        x_embedding = torch.arange(256, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)/256.0 # 1, 1, 1, 256
        y_embedding = torch.arange(256, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1)/256.0 # 1, 1, 256, 1
        x_embedding = x_embedding.repeat(1, 1, 256, 1) # 1, 256, 256, 1
        y_embedding = y_embedding.repeat(1, 1, 1, 256) # 1, 1, 256, 256
        xy_embedding = torch.cat([x_embedding, y_embedding], dim=1) # 1, 2, 256, 256
        

        self.position_embedding = nn.Parameter(xy_embedding, requires_grad=False)
        # Define the classification, regression, and centerness heads
        in_channels = 256
        # 分类的head, 用于判断当前位置，是否包含目标； 后面会添加sigmoid，然后预测IoU的值 ----> 可以得到概率分布图
        # 目标anchor的大小选择: 原图每个位置可以被覆盖两次； (每个位置的感受野，实际上是大于anchor的大小的)
        # sigmoid不利于部署，因此在计算图缩放时，使用sigmoid前的logits来实现；

        # self.rpn_head = nn.Conv2d(in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        
        # 回归的head, 用于预测目标的边界框位置
        # 用的是softmax的输出，乘以一个stride数，得到最终的边界位置
        # 方案一: nanodet的方案。在缩放后的feature map上，从中心向四周扩散，用softmax计算最终输出； --->softmax无法去除，不利于部署
        # 方案二: 回归x,y,w,h? 不符合直觉，因为我们不会去判断中心点在哪里，而是判断边界框的位置；
        # 方案三: 分别回归上下左右边界的偏移量，符合预期； 先直接用smooth-l1 loss来计算；
        #        直接回归x1,x2,y1,y2的normalized的值，效果不好；
        
        # 5个输出，分别预测每个位置，上下左右位置的相对值(0-1之间)，最后一个用于计算IoU loss；
        self.reg_head = nn.Conv2d(in_channels, self.num_classes * 4, kernel_size=3, stride=1, padding=1)
        self.cls_head = nn.Conv2d(in_channels, self.num_classes * 1, kernel_size=3, stride=1, padding=1)

        # TODO: 使用setcriterion来计算loss，从而去掉NMS过程； ---> 先考虑密集检测
        # TODO: 使用multi-level的输出； 先用分类score map逐级refine，最后在最大的feature map上进行回归；

    def forward(self, x):
        # Forward pass through the backbone network
        # Use the outputs to calculate class, regression, and centerness predictions
        # import numpy as np
        # print(np.shape(x), np.shape(self.position_embedding)) # torch.Size([8, 3, 256, 256]) torch.Size([1, 2, 256, 256])
        B = x.size(0)
        position_embedding = self.position_embedding.repeat((B, 1, 1, 1))
        x = torch.cat([x, position_embedding], dim=1) 
        
        x4 = self.backbone_stride4(x)
        
        x16 = self.backbone_stride16(x4)
        
        # import numpy as np
        # print("output feat shape", np.shape(x4), np.shape(x16))

        # rpn_logits = self.rpn_head(x16) # 16x16x4 ----> 暂时用不
        
        # scaled_x4 = self.get_scale_codinates(score_logits, x4) # 对原图进行缩放, 缩放到x16的大小
        
        # # 然后用score_logits对更大的feature map进行缩放； ----> 这里直接使用128x128的feature map来进行缩放；
        # x4_outputs = self.reg_head(scaled_x4) # TODO: 用score logits对x4进行缩放；这里不能和x16进行concat，二者语义是完全不同的，不存在对应关系；
        x16_reg_logits = self.reg_head(x16)
        x16_cls_logits = self.cls_head(x16)
        return rpn_logits, x16_reg_logits, x16_cls_logits # , x4_outputs

    def loss(self, rpn_logits, x16_reg_logits, x16_cls_logits, targets):
        
        rpn_targets, reg_targets, reg_mask, bboxes = targets

        # 分类IoU的loss
        eps = 1e-5
        preds = torch.clamp(torch.sigmoid(rpn_logits), eps, 1 - eps) # sigmoide获取概率
        alpha = 0.5
        gamma = 2
        loss_rpn = - alpha * (1 - preds) ** gamma * rpn_targets * torch.log(preds) - (1 - alpha) * preds ** gamma * (1 - rpn_targets) * torch.log(1 - preds)
        loss_rpn = torch.sum(loss_rpn)
        print("rpn:")
        print(preds)
        print(rpn_targets)
        exit(0)
        # reg的loss
        preds = torch.clamp(torch.sigmoid(x16_reg_logits), eps, 1 - eps) # B, num_cls * 4, 16, 16
        alpha = 0.5
        gamma = 2
        loss_reg = - alpha * (1 - preds) ** gamma * reg_targets * torch.log(preds) - (1 - alpha) * preds ** gamma * (1 - reg_targets) * torch.log(1 - preds)
        loss_reg = torch.sum(loss_reg * reg_mask)

        # iou预测loss
        # bboxes = preds * 256.0 # B, num_cls, 16, 16
        # preds = torch.sigmoid(x16_cls_logits) # B, num_cls * 4, 16, 16
        loss_dict = dict(
            loss_rpn = loss_rpn,
            loss_reg = loss_reg,
            loss = loss_reg + loss_rpn
        )
        return loss_dict
