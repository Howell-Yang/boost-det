import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
import numpy as np
import cv2
from datasets import ShapeDataset
from models import FCOS
from torch.utils.data import DataLoader


# 训练循环
def train(model, optimizer, dataloader):
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        # 将图像传入模型获取预测结果
        x16_cls_logits, x16_reg_logits = model(images)
        
        # 计算损失
        loss = model.loss(x16_cls_logits, x16_reg_logits, targets)
        
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

    # mean_ap = np.mean(ap_scores)
    # print(f"mAP: {mean_ap}")


def collate_fn(batch):
    # [[{'bbox': [141, 41, 180, 80], 'category': 'trapezoid'}], [{'bbox': [239, 168, 289, 218], 'category': 'trapezoid'}], [{'bbox': [237, 60, 265, 88], 'category': 'circle'}], [{'bbox': [86, 140, 163, 217], 'category': 'trapezoid'}, {'bbox': [39, 125, 115, 201], 'category': 'circle'}], [{'bbox': [165, 62, 202, 99], 'category': 'triangle'}, {'bbox': [203, 164, 258, 219], 'category': 'trapezoid'}, {'bbox': [198, 190, 230, 222], 'category': 'circle'}, {'bbox': [198, 169, 274, 245], 'category': 'triangle'}], [{'bbox': [183, 82, 228, 127], 'category': 'circle'}, {'bbox': [195, 198, 266, 269], 'category': 'circle'}], [{'bbox': [152, 123, 184, 155], 'category': 'trapezoid'}, {'bbox': [24, 53, 103, 132], 'category': 'square'}], [{'bbox': [174, 194, 241, 261], 'category': 'triangle'}]]
    # print(targets)
    #  [{k: v.to(device) for k, v in target.items()} for target in targets]
    images = []
    labels = []
    for i, (image, targets) in enumerate(batch):
        images.append(image)
        labels.append(targets)
    images = torch.stack(images)
    return images, labels

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
    # evaluate(model, eval_dataloader)
