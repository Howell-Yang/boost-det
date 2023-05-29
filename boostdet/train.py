import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from torchvision.ops import box_iou
import numpy as np
import cv2
from datasets import ShapeDataset
from models import FCOS
from torch.utils.data import DataLoader


# 训练循环
def train(model, optimizer, dataloader):
    model.train().cuda()
    for batch, (images, targets) in enumerate(dataloader):
        

        optimizer.zero_grad()
        
        # 将图像传入模型获取预测结果
        rpn_logits, x16_reg_logits, x16_cls_logits = model(images)
        
        # 计算损失
        loss = model.loss(rpn_logits, x16_reg_logits, x16_cls_logits, targets)
        
        # 反向传播和优化
        loss["loss"].backward()
        optimizer.step()

        # 打印损失值
        if batch % (len(dataloader)//10) == 0:
            print(f"{batch} : LR = {lr_scheduler.get_last_lr()}, Training loss:")
            for key, value in loss.items():
                print(f"{key} : {value.item()}")

    # 更新学习率
    lr_scheduler.step()

# 评估循环
def evaluate(model, dataloader):
    model.eval()    
    for batch, (images, targets) in enumerate(dataloader):
        # 将图像传入模型获取预测结果
        rpn_logits, reg_logits, cls_logits = model(images)
        loss = model.loss(rpn_logits, reg_logits, cls_logits, targets)
        # 打印损失值
        if batch % (len(dataloader)//10) == 0:
            print(f"{batch} : Evaluation loss:")
            for key, value in loss.items():
                print(f"{key} : {value.item()}")

            # 可视化部分
            bboxes = targets[-1]
            for i in range(len(images)):
                image = 255 * images[i].permute(1, 2, 0).contiguous().cpu().numpy()
                # colors = [(220,20,60), (255,0,255), (75,0,130), (30,144,255), (47,79,79)]
                colors = [(0, 0, 255)] * 10
                # GT
                for bbox in bboxes[i]:
                    bbox = bbox.cpu().numpy()[0]
                    cat_idx = int(bbox[4])
                    x1, y1, x2, y2 = list(map(int, bbox[:4]))
                    # print(np.shape(image), np.max(image))
                    cv2.rectangle(image, (x1, y1), (x2, y2), colors[cat_idx], 1)

                # PD
                colors = [(0, 0, 0)] * 10
                for cat_idx in range(num_classes):
                    score_mask = rpn_logits[i, cat_idx, :, :].sigmoid()
                    reg_mask = reg_logits[i, cat_idx * num_classes: (cat_idx + 1) * num_classes, :, :].sigmoid()
                    score_mask = score_mask.reshape(-1,)
                    reg_mask = reg_mask.reshape(-1, 4)
                    normed_bboxes = reg_mask[score_mask > 0.1]
                    for normed_bbox in normed_bboxes.detach().cpu().numpy():
                        x1, y1, x2, y2 = list(map(int, normed_bbox * 256))
                        cv2.rectangle(image, (x1, y1), (x2, y2), colors[cat_idx], 1)
                cv2.imwrite("./image_{}_{}.png".format(batch, i), image)
    # # 计算mAP
    # gt_boxes = np.concatenate(gt_boxes, axis=0)
    # pred_boxes = np.concatenate(pred_boxes, axis=0)
    # pred_scores = np.concatenate(pred_scores, axis=0)
    # pred_labels = np.concatenate(pred_labels, axis=0)
    # gt_labels = np.concatenate(gt_labels, axis=0)
    # num_classes = model.num_classes
    # ap_scores = []
    
    # for class_id in range(1, num_classes):  # 类别从1开始，不包括背景类别
    #     gt_class_boxes = gt_boxes[gt_labels == class_id]
    #     pred_class_boxes = pred_boxes[pred_labels == class_id]
    #     pred_class_scores = pred_scores[pred_labels == class_id]
    #     ap = average_precision_score(gt_class_boxes, pred_class_scores, pred_class_boxes)
    #     ap_scores.append(ap)
    #     print(f"Class {class_id} AP: {ap}")
    
    # mean_ap = np.mean(ap_scores)
    # print(f"mAP: {mean_ap}")


def collate_fn(batch):
    # [[{'bbox': [141, 41, 180, 80], 'category': 'trapezoid'}], [{'bbox': [239, 168, 289, 218], 'category': 'trapezoid'}], [{'bbox': [237, 60, 265, 88], 'category': 'circle'}], [{'bbox': [86, 140, 163, 217], 'category': 'trapezoid'}, {'bbox': [39, 125, 115, 201], 'category': 'circle'}], [{'bbox': [165, 62, 202, 99], 'category': 'triangle'}, {'bbox': [203, 164, 258, 219], 'category': 'trapezoid'}, {'bbox': [198, 190, 230, 222], 'category': 'circle'}, {'bbox': [198, 169, 274, 245], 'category': 'triangle'}], [{'bbox': [183, 82, 228, 127], 'category': 'circle'}, {'bbox': [195, 198, 266, 269], 'category': 'circle'}], [{'bbox': [152, 123, 184, 155], 'category': 'trapezoid'}, {'bbox': [24, 53, 103, 132], 'category': 'square'}], [{'bbox': [174, 194, 241, 261], 'category': 'triangle'}]]
    # print(targets)
    #  [{k: v.to(device) for k, v in target.items()} for target in targets]
    device = "cuda" # "cuda"
    N = len(batch) # 多个batch的list
    stride = 16
    rpn_targets = torch.zeros((N, num_classes, 16, 16)).to(device) # 在16x16的feature map上做anchor的IoU计算
    reg_targets = torch.zeros((N, num_classes * 4, 16, 16)).to(device)
    reg_mask = torch.zeros((N, num_classes * 4, 16, 16)).to(device)
    images = []
    bboxes = []
    for i, (image, targets) in enumerate(batch):
        images.append(image)
        rpn_x = torch.arange(0, 16, dtype=torch.float32).unsqueeze(1).repeat((1, 16)).to(device) * stride # 16, 16
        rpn_y = torch.arange(0, 16, dtype=torch.float32).unsqueeze(0).repeat((16, 1)).to(device) * stride # 16, 16        
        rpn_w = torch.ones_like(rpn_x, dtype=torch.float32).to(device) * stride * 2 # 16x16x1
        rpn_h = torch.ones_like(rpn_y, dtype=torch.float32).to(device) * stride * 2  # 16x16x1
        rpn_bboxes = torch.stack([rpn_x - rpn_w/2.0, rpn_y - rpn_h/2.0, rpn_x + rpn_w/2.0, rpn_y + rpn_h/2.0], dim=-1) # 4, 16, 16
        rpn_bboxes = rpn_bboxes.reshape(-1, 4) # 256, 4
        single_bboxes = []
        for cat_idx in range(num_classes):
            reg_targets[i, cat_idx*num_classes + 2: (cat_idx + 1)*num_classes, :, :] = 1.0
            target_bboxes = []
            for target in targets: # 这里是一个list [{'bbox': [141, 41, 180, 80], 'category': 'trapezoid'}]
                bbox = target['bbox'] # x1, x2, y1, y2
                category = shapes2label[target['category']]
                if category == cat_idx:
                    target_bboxes.append(bbox + [cat_idx])
            if len(target_bboxes) == 0:
                continue
            target_bboxes = torch.tensor(target_bboxes, dtype=torch.float32).to(device).reshape(-1, 5) # M, 4
            single_bboxes.append(target_bboxes)

            ious = box_iou(rpn_bboxes, target_bboxes[:, :4]) # M, N
            # print(np.shape(ious))
            max_ious, iou_index = torch.max(ious, dim=-1)
            # print("rpn anchor(x1, y1, x2, y2)", rpn_bboxes[16 * 16 //2,:])
            # print("rpn anchor(x1, y1, x2, y2)", rpn_bboxes[16 * 16 //2 + 1,:])
            # cv2.imwrite("./IoU_sample.png", image.permute(1, 2, 0).contiguous().cpu().numpy() * 255.0)
            # print(max_ious)
            # iou_map = max_ious.reshape(16, 16).cpu().numpy()  * 255.0
            # print(iou_map)
            # cv2.imwrite("./IoU_map.png", iou_map)
            # exit(0)
            rpn_targets[i, cat_idx, :, :] = max_ious.reshape(16, 16) # 16, 16
            # reg to normalized cordinates ---> 每个位置，都有一个regression targets
            reg_bboxes = target_bboxes[iou_index, :4].view(16, 16, 4).permute([2, 0, 1])/256.0 # 每个位置，都有一个regression targets
            pos_index = max_ious.reshape(16, 16) > 0.00001
            reg_targets[i, cat_idx*num_classes: (cat_idx+1)*num_classes, :, :] = reg_bboxes # 0-256 ---> 0-1
            reg_mask[i, cat_idx*num_classes: (cat_idx+1)*num_classes, :, :] = pos_index.unsqueeze(0).repeat((num_classes, 1, 1))
        bboxes.append(single_bboxes)
    images = torch.stack(images).to(device)
    return images, (rpn_targets, reg_targets, reg_mask, bboxes)

# 参数定义
image_size = (256, 256)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
shapes = ['square', 'circle', 'triangle', 'trapezoid']
shapes2label = {'square': 0, 'circle': 1, 'triangle': 2, 'trapezoid': 3}
num_classes = len(shapes)
batch_size = 8
num_workers = 0 # 4
num_epochs = 100
max_lr = 1e-3


# 创建ShapeDataset实例
num_samples = 16000
train_dataset = ShapeDataset(image_size, num_samples, colors, shapes)
val_dataset = ShapeDataset(image_size, 160, colors, shapes)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

# 创建模型实例&定义优化器和学习率调度器
model = FCOS(num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=num_epochs, steps_per_epoch=len(train_dataloader), verbose=True)

# 训练模型
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, optimizer, train_dataloader)
    evaluate(model, eval_dataloader)
