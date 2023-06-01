import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import os

class ShapeDataset(Dataset):
    def __init__(self, image_size, num_samples, colors, shapes):
        self.image_size = image_size
        self.num_samples = num_samples
        self.colors = colors
        self.shapes = shapes
        self.shapes2label = {shape: i for i, shape in enumerate(self.shapes)}
        self.count_index = 0

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        image = Image.new("RGB", self.image_size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        num_objects = np.random.randint(1, 5)  # 随机生成目标数量
        
        labels = []
        
        for _ in range(num_objects):
            # 随机选择一个颜色和形状
            color = np.random.choice(range(len(self.colors)))
            color = self.colors[color]
            shape = np.random.choice(self.shapes)
            
            # 随机生成形状的参数
            size = np.random.randint(20, 80)
            x = np.random.randint(size, self.image_size[0] - size)
            y = np.random.randint(size, self.image_size[1] - size)
            shape_params = [x, y, x + size, y + size]

            # 绘制形状
            if shape == 'square':
                draw.rectangle(shape_params, fill=color)
            elif shape == 'circle':
                draw.ellipse(shape_params, fill=color)
            elif shape == 'triangle':
                points = [(x, y), (x + size, y), (x + size// 2, y + size)]
                draw.polygon(points, fill=color)
            elif shape == 'trapezoid':
                points = [(x, y), (x + size, y), (x + size * 3 // 4, y + size), (x + size // 4, y + size)]
                draw.polygon(points, fill=color)
            
            # 构建标签
            label = {
                'bbox': shape_params,
                'category': self.shapes2label[shape],
            }
            
            labels.append(label)
        
        # 可视化图像
        # import cv2
        # draw_image = np.array(image)
        # for label in labels:
        #     print(label['bbox'])
        #     cv2.rectangle(draw_image, (label['bbox'][0], label['bbox'][1]), (label['bbox'][2], label['bbox'][3]), (0, 0, 0), 1)
        # cv2.imwrite("dataset_sample_{}.jpg".format(self.count_index), draw_image)
        self.count_index += 1
    
        # 将图像转换为张量并进行归一化
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image, labels


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    image_size = (256, 256)
    num_samples = 100
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    shapes = ['square', 'circle', 'triangle', 'trapezoid']
    train_dataset = ShapeDataset(image_size, num_samples, colors, shapes)
    val_dataset = ShapeDataset(image_size, 160, colors, shapes)
    batch_size = 10
    num_workers = 0
    def collate_fn(batch):
        images = []
        for i, (image, targets) in enumerate(batch):
            images.append(image)
        images = torch.stack(images)
        return images, images
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    for image, target in train_dataloader:
        continue    
