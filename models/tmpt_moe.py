
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
import numpy as np


class Expert(nn.Module):
    def __init__(self, linear_injection=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(linear_injection * 2, 2), nn.Sigmoid())
    def forward(self, text_features,visual_features):
        concat_feat = torch.cat([text_features, visual_features], dim=-1)
        gate_value = self.gate(concat_feat)
        # print(gate_value)
        concat_feat = torch.cat([gate_value[:, 0:1] * text_features, gate_value[:, 1:2] * visual_features],dim=-1)
        return concat_feat,gate_value



class TopKRouter(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int =1024 ,capacity_factor: float = 1.0):
        super().__init__()
        # self.top_k = k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        # 门控网络：一个线性层，输出维度为专家数量
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x: torch.Tensor,top_k: int = 2):
        logits = self.gate(x)  # 形状: [batch_size, seq_len, num_experts]
        # 计算TopK：得到分数值和索引
        top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)  # 形状均为: [batch_size, seq_len, top_k]
        # print(top_k_logits,top_k_indices)
        # 在TopK分数上计算Softmax，得到归一化权重
        probs = F.softmax(top_k_logits, dim=-1)  # 形状: [batch_size, seq_len, top_k]
        # 计算负载均衡损失（简化版）
        # 思路：鼓励每个专家被选中的概率分布均匀
        gates_for_loss = F.softmax(logits, dim=-1)  # 在全部专家上计算softmax
        mean_expert_gate = gates_for_loss.mean(dim=0)  # 每个专家的平均激活概率 [num_experts]
        load_balance_loss = -(mean_expert_gate * torch.log(mean_expert_gate + 1e-6)).sum()
        # gates = F.softmax(logits, dim=-1)  # 所有专家的门控值 [n_tokens, num_experts]
        # mean_gate_per_expert = gates.mean(dim=0)  # [num_experts]
        # target = torch.ones_like(mean_gate_per_expert) / self.num_experts
        # load_balance_loss = F.mse_loss(mean_gate_per_expert, target)
        return probs, top_k_indices, load_balance_loss
from models.miloss import MultiModalMIAlignmentLoss
class TMPTModelVisualGateMOEMiloss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tmpt_textual_model = TMPTTextualModel(args)
        self.tmpt_visual_model = TMPTVisualModel(args)
        for k, p in self.tmpt_visual_model.named_parameters():
            if "prompt" not in k and "pooler" not in k:
                p.requires_grad = False
        if args.linear_injection == -1:
            linear_injection = min(self.tmpt_textual_model.hidden_size, self.tmpt_visual_model.hidden_size)
        else:
            linear_injection = args.linear_injection

        self.textual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.tmpt_textual_model.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))

        self.visual_transformer_linear = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(self.tmpt_visual_model.hidden_size, linear_injection)),
            ('layerNorm', nn.BatchNorm1d(linear_injection)),
            ('activate', nn.LeakyReLU(0.2))
        ]))
        self.stance_loss_fn = nn.CrossEntropyLoss()

        self.stance_classifier = nn.Linear(linear_injection*2 , args.label_size)

        self.num_experts = args.num_experts
        # self.top_k = args.top_k#args.num_experts
        self.capacity_factor = 1  # 容量因子，用于负载均衡
        self.router = TopKRouter(num_experts=args.num_experts,capacity_factor=self.capacity_factor)
        self.experts = nn.ModuleList(
            [Expert(linear_injection) for _ in range(args.num_experts)]
        )

        self.mi_loss_fn = MultiModalMIAlignmentLoss({
            'text_dim': linear_injection,
            'image_dim': linear_injection,
            'temperature': 0.07,
            'mi_estimator': 'infonce',#infonce',  # 或 'clip' js
            'alpha': 1.0,  # 模态间MI权重
            'beta_text': 1.0,  # 文本-标签MI权重
            'beta_image': 1.0,  # 图像-标签MI权重
            'use_projection': True
        })
        self.alpha = nn.Parameter(torch.randn(args.num_experts))  # 对应4个专家的对齐损失系数，随机初始化
        self.beta = nn.Parameter(torch.randn(args.num_experts))  # 对应4个专家的I(T,Y)系数，随机初始化
        self.gamma = nn.Parameter(torch.randn(args.num_experts))  # 对应4个专家的I(V,Y)系数，随机初始化

    def forward(self, input_data,top_k: int = 2):
        # print(top_k)
        textual_outputs = self.tmpt_textual_model(input_data)
        text_pooled_output = self.textual_transformer_linear(textual_outputs['last_hidden_state'])

        visual_outputs = self.tmpt_visual_model(input_data)
        image_pooled_output = self.visual_transformer_linear(visual_outputs['last_hidden_state'])
        #1)直接拼接
        concat_feat = torch.cat([text_pooled_output, image_pooled_output], dim=-1)
        router_weights,expert_indices, moe_loss=self.router(concat_feat,top_k=top_k)
        # print(router_weights,expert_indices, moe_loss)
        batch_size = text_pooled_output.shape[0]
        expert_outputs = []
        expert_gate=[]
        for i,expert in enumerate(self.experts):
            # 专家需要处理batch中所有样本
            expert_out,gate_value = expert(text_pooled_output, image_pooled_output)  # [batch_size, hidden_dim]
            expert_outputs.append(expert_out)
            expert_gate.append(gate_value)
        outputs = torch.stack(expert_outputs, dim=-1)
        full_weights = torch.zeros(batch_size, self.num_experts,device=router_weights.device)
        for i in range(batch_size):
            full_weights[i, expert_indices[i]] = router_weights[i]
        concat_feat = (outputs * full_weights.unsqueeze(1)).sum(dim=-1)
        # print(concat_feat.shape)
        stance_logits = self.stance_classifier(concat_feat)
        stance_loss = self.stance_loss_fn(stance_logits, input_data['stance_label'])
        # print(stance_loss)
        mi_loss, mi_loss_dict = self.mi_loss_fn(text_pooled_output, image_pooled_output, input_data['stance_label'])
        # 步骤1：对可学习参数做Sigmoid，约束在[0,1]，得到α_k, β_k, γ_k
        a_k = torch.sigmoid(self.alpha)  # [4,] 学出来的专家对齐系数
        b_k = torch.sigmoid(self.beta)  # [4,] 学出来的专家I(T,Y)系数
        c_k = torch.sigmoid(self.gamma)  # [4,] 学出来的专家I(V,Y)系数
        # 步骤2：加权求和，得到动态a,b,c（和你之前的公式完全一样！）
        a = torch.matmul(full_weights, a_k)  # [batch_size,] 动态对齐损失权重
        b = torch.matmul(full_weights, b_k)  # [batch_size,] 动态I(T,Y)权重
        c = torch.matmul(full_weights, c_k)  # [batch_size,] 动态I(V,Y)权重
        mi_loss=(a*mi_loss_dict['mi_inter']).mean()+(b*mi_loss_dict['mi_text_label']).mean() + (c*mi_loss_dict['mi_image_label']).mean()
        # print(mi_loss)
        stance_loss = stance_loss  + 0.1*mi_loss + 0.01 * moe_loss
        # print(stance_loss)
        return stance_logits, stance_loss

