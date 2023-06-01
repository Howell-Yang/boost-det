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
from utils import distance2bbox


# 训练循环
def train(model, optimizer, dataloader):
    model.train()
    for batch, (images, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        # 将图像传入模型获取预测结果
        x16_cls_logits, x16_reg_logits = model(
            images.to(model.cls_head.weight.device))

        # 计算损失
        loss = model.loss(x16_cls_logits, x16_reg_logits, targets)

        # 反向传播和优化
        loss["loss"].backward()
        optimizer.step()

        # 打印损失值
        if batch % (max(1, len(dataloader) // 10)) == 0:
            print(
                f"{batch} : LR = {lr_scheduler.get_last_lr()}, Training loss:")
            for key, value in loss.items():
                print(f"{key} : {value.item()}")
            print("target scale:", model.gfl_loss.regression_scale,
                  model.gfl_loss.regression_scale.grad)

    # 更新学习率
    lr_scheduler.step()


# 评估循环
def evaluate(model, dataloader):
    model.eval()
    for batch, (images, targets) in enumerate(dataloader):
        # 将图像传入模型获取预测结果
        x16_cls_logits, x16_reg_logits = model(
            images.to(model.cls_head.weight.device))

        # 计算损失
        loss = model.loss(x16_cls_logits, x16_reg_logits, targets)
        N, C, H, W = x16_reg_logits.size()
        x16_cls_logits = x16_cls_logits.permute(0, 2, 3, 1).contiguous() # N, H, W, C
        x16_reg_logits = x16_reg_logits.permute(0, 2, 3, 1).contiguous().view(N, H, W, model.num_classes, 4 * (2 * model.reg_max + 1))
        # 打印损失值
        if batch % (max(1, len(dataloader) // 10)) == 0:
            print(f"{batch} : Evaluation loss:")
            for key, value in loss.items():
                print(f"{key} : {value.item()}")

            # 可视化部分
            for i in range(len(images)):
                image = 255 * images[i].permute(1, 2,
                                                0).contiguous().cpu().numpy()
                # colors = [(220,20,60), (255,0,255), (75,0,130), (30,144,255), (47,79,79)]
                colors = [(i * 10, 255, 0) for i in range(10)]
                # GT
                label_info =  targets[i]
                for bbox_info in label_info:
                    x1, y1, x2, y2 = bbox_info['bbox']
                    cat_idx = bbox_info['category']
                    x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
                    # print(np.shape(image), np.max(image))
                    cv2.rectangle(image, (x1, y1), (x2, y2), colors[cat_idx], 1) # GT is green

                # PD
                colors = [(0, i * 10,  255) for i in range(10)]
                for cat_idx in range(num_classes):
                    # decode each class
                    scores = x16_cls_logits[i, :, :, cat_idx].sigmoid() # sigmoid 得分 ----> 对应的是IOU分数
                    corners = x16_reg_logits[i, :, :, cat_idx, :] # H,W,15x4
                    pos_mask = scores.view(-1) > 0.5 # 选出得分大于0.1的位置
                    corners = corners.view(-1, model.num_classes * (2 * model.reg_max + 1)) # Nx4 x 15
                    # print(np.shape(pos_mask))
                    pos_corner_pred = corners[pos_mask]
                    # print(np.shape(pos_corner_pred))
                    pos_corner_pred = model.gfl_loss.distribution_project(pos_corner_pred) # N x 4
                    
                    stride = 16
                    x_coord = torch.arange(
                    image_size[0] // stride,
                    dtype=torch.int32,
                    device=model.gfl_loss.regression_scale.device).unsqueeze(1)  # 32 x 1
                    y_coord = torch.arange(
                        image_size[1] // stride,
                        dtype=torch.int32,
                        device=model.gfl_loss.regression_scale.device).unsqueeze(0)  # 1 x 32
                    x_coord = x_coord.expand(-1, image_size[1] //
                                            stride)  # 32 x 32 ---> x[0][0] != x[1][0]
                    y_coord = y_coord.expand(image_size[0] // stride, -1)
                    xy_coord = torch.stack([x_coord, y_coord], dim=2)  # 32 x 32 x 2
                    xy_coord = xy_coord.unsqueeze(0)

                    grid_centers = xy_coord.reshape(-1, 2)  # x, y 中心点坐标
                    pos_grid_centers = grid_centers[pos_mask]
                    
                    # print(pos_grid_centers)
                    # print(pos_corner_pred)
                    pos_corner_scale = model.gfl_loss.regression_scale[0, cat_idx, :]
                    pos_corner_pred = pos_corner_pred / pos_corner_scale
                    # print("pos_corner_pred", pos_corner_pred)
                    pos_decode_bbox_pred = distance2bbox( # 这里需要处理regression_scale, 按照类别进行scale
                        pos_grid_centers,
                        pos_corner_pred,
                        max_shape=[image_size[0] // stride, image_size[1] // stride]  # x1 = x1.clamp(min=0, max=max_shape[1])
                    )
                    # print("pos_corner_pred", pos_corner_pred)
                    pos_decode_bbox_pred = pos_decode_bbox_pred * stride
                    for bbox in pos_decode_bbox_pred.detach().cpu().numpy():
                        x1, y1, x2, y2 = list(map(int, bbox))
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
batch_size = 16
num_workers = 0  # 4
num_epochs = 10
max_lr = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建ShapeDataset实例
num_samples = 16000
train_dataset = ShapeDataset(image_size, num_samples, colors, shapes)
val_dataset = ShapeDataset(image_size, 16, colors, shapes)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
eval_dataloader = DataLoader(val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=collate_fn)

# 创建模型实例&定义优化器和学习率调度器
model = FCOS(num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                            lr=max_lr,
                            # momentum=0.9,
                            weight_decay=0.0001)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr,
    total_steps=None,
    epochs=num_epochs,
    steps_per_epoch=len(train_dataloader),
    verbose=True)

# 训练模型
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, optimizer, train_dataloader)
    evaluate(model, eval_dataloader)
