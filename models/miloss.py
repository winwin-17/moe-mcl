from torch.nn import Dropout
from typing import List, Tuple, Dict
import math
import inspect
from torch.nn import Dropout
from transformers import AutoConfig, AutoModel
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tmpt import TMPTTextualModel,TMPTVisualModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiModalMIAlignmentLoss1(nn.Module):
    """
    多模态互信息对齐损失
    目标：
    1. 最大化模态间互信息（text-image对齐）
    2. 最大化单模态与标签的互信息（任务相关最大化）
    3. 可选的：最小化模态间冗余信息
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 温度参数
        self.temperature = config.get('temperature', 0.07)

        # 互信息估计方法
        self.mi_estimator = config.get('mi_estimator', 'infonce')  # 'infonce', 'js', 'clip'

        # 权重系数
        self.alpha = config.get('alpha', 1.0)  # 模态间MI权重
        self.beta_text = config.get('beta_text', 0.5)  # 文本-标签MI权重
        self.beta_image = config.get('beta_image', 0.5)  # 图像-标签MI权重

        # 是否使用投影头
        self.use_projection = config.get('use_projection', True)

        if self.use_projection:
            # 投影到低维空间进行对比学习
            self.text_projection = nn.Sequential(
                nn.Linear(config.get('text_dim', 768), 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )

            self.image_projection = nn.Sequential(
                nn.Linear(config.get('image_dim', 512), 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )

        # 判别器网络（用于JS MI估计）
        if self.mi_estimator == 'js':
            input_dim = config.get('text_dim', 768) + config.get('image_dim', 512)
            self.discriminator = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

    def forward(self, text_features, image_features, labels=None):
        """
        计算互信息对齐损失

        Args:
            text_features: 文本特征 [batch_size, text_dim]
            image_features: 图像特征 [batch_size, image_dim]
            labels: 标签 [batch_size] 或 [batch_size, 1]

        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量
        """
        loss_dict = {}

        # 1. 投影到对比学习空间
        if self.use_projection:
            text_proj = self.text_projection(text_features)
            image_proj = self.image_projection(image_features)
        else:
            text_proj = F.normalize(text_features, dim=-1)
            image_proj = F.normalize(image_features, dim=-1)


        if self.mi_estimator == 'infonce':
            mi_inter_loss = self._infonce_loss(text_proj, image_proj)
        elif self.mi_estimator == 'clip':
            mi_inter_loss = self._clip_contrastive_loss(text_proj, image_proj)
        elif self.mi_estimator == 'js':
            mi_inter_loss = self._js_mi_loss(text_proj, image_proj, mode='maximize')
        else:
            raise ValueError(f"Unknown MI estimator: {self.mi_estimator}")

        loss_dict['mi_inter'] = mi_inter_loss.item()
        mi_inter_loss = self.alpha * mi_inter_loss

        # 3. 计算单模态与标签的互信息（任务相关最大化）
        if labels is not None and (self.beta_text > 0 or self.beta_image > 0):
            # 确保标签格式正确
            if labels.dim() == 1:
                labels = labels.view(-1, 1).float()

            # 文本-标签互信息
            if self.beta_text > 0:
                mi_text_label = self._supervised_mi_loss(text_proj, labels)
                loss_dict['mi_text_label'] = mi_text_label.item()
                mi_text_label = self.beta_text * mi_text_label

            # 图像-标签互信息
            if self.beta_image > 0:
                mi_image_label = self._supervised_mi_loss(image_proj, labels)
                loss_dict['mi_image_label'] = mi_image_label.item()
                mi_image_label = self.beta_image * mi_image_label

        # 4. 计算总损失
        total_loss = mi_inter_loss
        if self.beta_text > 0:
            total_loss += mi_text_label
        if self.beta_image > 0:
            total_loss += mi_image_label

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def _infonce_loss(self, z1, z2):
        """InfoNCE损失 - 最大化正样本对的互信息"""
        batch_size = z1.shape[0]

        # 归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.T) / self.temperature

        # 创建标签
        labels = torch.arange(batch_size, device=z1.device)

        # 对称损失
        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_t = F.cross_entropy(sim_matrix.T, labels)

        return (loss_i + loss_t) / 2

    def _clip_contrastive_loss(self, image_features, text_features):
        """CLIP风格的对比损失"""
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = text_features @ image_features.T / self.temperature

        # 创建标签
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # 计算损失
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        return (loss_i + loss_t) / 2

    def _js_mi_loss(self, x, y, mode='maximize'):
        """Jensen-Shannon互信息估计损失"""
        batch_size = x.shape[0]

        # 正样本对
        pos_pairs = torch.cat([x, y], dim=1)
        pos_score = self.discriminator(pos_pairs)

        # 负样本对（shuffle y）
        perm = torch.randperm(batch_size)
        y_shuffled = y[perm]
        neg_pairs = torch.cat([x, y_shuffled], dim=1)
        neg_score = self.discriminator(neg_pairs)

        # JS MI估计
        ej = -F.softplus(-pos_score).mean()  # E[log(1 + e^{-T})]
        em = F.softplus(neg_score).mean()  # E[log(1 + e^{T})]

        mi_estimate = ej - em

        if mode == 'maximize':
            return -mi_estimate  # 负值用于最大化
        else:
            return mi_estimate  # 正值用于最小化

    def _supervised_mi_loss(self, features, labels):
        """
        监督互信息损失 - 最大化特征与标签的互信息
        使用标签感知的对比学习
        """
        batch_size = features.shape[0]

        # 归一化特征
        features = F.normalize(features, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.T) / self.temperature

        # 创建标签掩码（正样本对：相同标签）
        if labels.dim() == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()

        # 排除自身
        self_mask = torch.eye(batch_size, device=features.device)
        pos_mask = label_mask - self_mask

        # 计算正样本损失
        # 为了数值稳定性，减去最大值
        logits = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()

        # 计算对数概率
        exp_logits = torch.exp(logits) * (1 - self_mask)  # 排除自身
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 平均正样本的对数概率
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)

        # 损失（最大化正样本的对数概率 -> 最小化负值）
        loss = -mean_log_prob_pos.mean()

        return loss