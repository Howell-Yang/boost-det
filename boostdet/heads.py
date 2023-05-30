import torch
import torch.nn as nn
from losses.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from losses.iou_loss import GIoULoss, bbox_overlaps
import torch.nn.functional as F


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x

class GeneralizedFocalLoss(nn.Module):
    def __init__(self, reg_max = 7, use_sigmoid=True, num_classes=4):
        # 模型输出(多个list): [x16_logits], [x16_reg];
        # 原始targets bbox: B, N, 5 动态shape
        self.reg_max = 7
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid
        self.regression_scale = torch.nn.Linear(self.num_classes, self.num_classes)
        self.distribution_project = Integral(self.reg_max)
        self.loss_qfl = QualityFocalLoss(
            use_sigmoid=self.use_sigmoid,
            beta=2,
            loss_weight=0.25,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=0.50
        )
        self.loss_bbox = GIoULoss(loss_weight=0.25)
    
    
    def assign_targets(self,
        label_infos, # bbox_targets: B, N, 5
        stride = 16,
        image_size = (256, 256),
        ):
        B = len(label_infos)
        feat_size = (image_size[0] // stride, image_size[1] // stride)
        cls_targets = []
        bbox_targets = []
        num_total_samples = 0
        for i, label_info in enumerate(label_infos):
            cls_target = torch.new_ones((feat_size[0], feat_size[1]), dtype=torch.long) * self.num_classes # H, W
            bbox_target = torch.new_zeros((feat_size[0], feat_size[1], self.num_classes,  4), dtype=torch.float32) # H, W, num_cls, 4
            for bbox_info in label_info:
                x1, y1, x2, y2 = bbox_info['bbox']
                category = bbox_info['category']
                cx = (x1 + x2)/2.0
                cy = (y1 + y2)/2.0
                # 每个bbox，会被分配给最近的点 ----> 一个点可能会分配多个bbox
                for offset_x in range(0, 2):
                    for offset_y in range(0, 2):
                        grid_cx = int(cx / stride) + offset_x
                        grid_cy = int(cy / stride) + offset_y
                        if grid_cx >= feat_size[0] or grid_cy >= feat_size[1]:
                            continue
                        if grid_cy < 0 or grid_cy < 0:
                            continue
                        cls_target[grid_cx, grid_cy] = category # 0 - num_cls - 1 for objects, num_cls for background
                        num_total_samples += 1

                        # 计算每个位置, 相对中心位置的偏移量 ---> 由于中心点不一定在bbox内，所以回归坐标是存在负值的情况的
                        left_offset = grid_cx - x1 / stride  # 上下左右坐标偏移量，从当前点，向边界点移动；符合直觉
                        right_offset = x2 / stride - grid_cx
                        top_offset = grid_cy - y1/stride
                        bottom_offset = y2/stride - grid_cy
                        bbox_target[grid_cx, grid_cy, category, 0] = left_offset
                        bbox_target[grid_cx, grid_cy, category, 1] = right_offset
                        bbox_target[grid_cx, grid_cy, category, 2] = top_offset
                        bbox_target[grid_cx, grid_cy, category, 3] = bottom_offset
                        
                        # 回归的目标是每个坐标线，相对中心点的偏移量
                        # 但是，中间还涉及到target层面的scale
                        # 如果target层面有可学习的target scale参数，能够更好地处理
            cls_targets.append(cls_target)
            bbox_targets.append(bbox_target)
        return cls_targets, bbox_targets, num_total_samples

    def loss_single(
        self,
        cls_score, # logits before sigmoid: N, num_classes, H, W
        bbox_pred, # logits before sigmoid: N, 4 * num_classes *(reg_max + 1), H, W
        cls_target, # N, H, W
        bbox_targets, # N, H, W, num_classes, 4
        stride,
        num_total_samples,
    ):

        # cls_score = cls_score.reshape(-1, self.num_classes)  # N, num_classes, H, W
        # bbox_pred = bbox_pred.reshape(-1, self.num_classes, 4 * (self.reg_max + 1)) # regrssion ---> 不同类别用的相同的回归，这里是不合理的
        bbox_pred = bbox_pred.permute([0, 2, 3, 1]).reshape(-1, self.num_classes, 4 * (self.reg_max + 1)) # N, C, H, W ---> N, H, W, C
        cls_score = cls_score.permute([0, 2, 3, 1]).reshape(-1, self.num_classes)

        cls_target = cls_target.reshape(-1,) # N x H x W --> NHW, -1 每个位置的类别index
        bbox_targets = bbox_targets.reshape(-1, self.num_classes, 4) # NHW, self.num_classes, 4
        bbox_targets = self.regression_scale(bbox_targets) # NHW, self.num_classes, 4

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes # 每个位置会分配一个类别Index
        pos_index = torch.nonzero(
            (cls_target >= 0) & (cls_target < bg_class_ind), as_tuple=False
        ).squeeze(1) # N x 1



        score = bbox_pred.new_zeros(bbox_targets.shape)

        if len(pos_index) > 0: # 有正样本
            # 回归时，只回归正样本的bbox坐标
            pos_bbox_targets = bbox_targets[pos_index] # (n, self.num_classes, 4) # 这里的regression target是相对于feature map的偏移量
            pos_bbox_pred = bbox_pred[pos_index]  # (n, self.num_classes, 4 * (reg_max + 1))

            # 按照类别映射，并按照类别进行scale ---> 根据target进行映射
            # pos_bbox_targets = torch.gather(pos_bbox_targets, dim=1, index=cls_target)
            # pos_bbox_pred = torch.gather(pos_bbox_pred, dim=1, index=cls_target)
            pos_bbox_targets=torch.index_select(pos_bbox_targets,dim=1,index=cls_target)
            pos_bbox_pred = torch.index_select(pos_bbox_pred, dim=1, index=cls_target)


            # 获取score预测值，作为weight
            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_index]


            # 计算IoU损失
            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(
                pos_grid_cell_centers, pos_bbox_pred_corners
            )
            pos_decode_bbox_targets = pos_bbox_targets / stride
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(
                pos_grid_cell_centers, pos_decode_bbox_targets, self.reg_max
            ).reshape(-1)

            # IoU loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0,
            )

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0,
            )
        else: # 没有正样本
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).to(cls_score.device)

        # qfl loss
        loss_qfl = self.loss_qfl(
            cls_score,
            (cls_target, score),
            weight=None,
            avg_factor=num_total_samples,
        )

        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()

    def loss(self, outputs, label_info):
        cls_score, bbox_pred = outputs # 假设只有一个level的输出
        stride = 16 # 这个level的stride
        image_zize = (256, 256)

        # 第一步，分配target
        cls_target, bbox_targets, num_total_samples = self.assign_targets(label_info, stride, image_zize)
        num_total_samples = max(num_total_samples, 1.0)

        # 第二步，计算每部分的loss
        loss_qfl, loss_bbox, loss_dfl, avg_factor = self.loss_single(
            cls_score, # logits before sigmoid: N, num_classes, H, W
            bbox_pred, # logits before sigmoid: N, 4 * num_classes *(reg_max + 1), H, W
            cls_target, #
            bbox_targets,
            stride,
            num_total_samples,
        )

        # 第三步，计算总loss
        avg_factor = sum(avg_factor).item()
        loss_bbox = loss_bbox / avg_factor  #IoU损失，与bbox个数有关
        loss_dfl = loss_dfl / avg_factor # DFL损失，与bbox个数有关
        loss = loss_qfl + loss_bbox + loss_dfl
        return loss