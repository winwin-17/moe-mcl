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

class TMPTVisualModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.config = AutoConfig.from_pretrained(args.visual_transformer_name)
        self.visual_plm = AutoModel.from_pretrained(args.visual_transformer_name, self.config)
        self.visual_soft_tokens = args.visual_soft_tokens
        self.hidden_size = self.config.hidden_size

        self.soft_prompt_dropout = Dropout(args.visual_soft_prompt_dropout)

        val = math.sqrt(6. / float(self.hidden_size * 2))
        self.soft_prompt_embeds = nn.Parameter(torch.zeros(1, self.visual_soft_tokens, self.hidden_size))
        # xavier_uniform initialization
        nn.init.uniform_(self.soft_prompt_embeds.data, -val, val)

    def incorporate_prompt(self, pixel_values):
        # combine prompt embeddings with text embeddings
        batch_size = pixel_values.shape[0]
        x = self.visual_plm.embeddings(pixel_values)  # (batch_size, sen len, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.soft_prompt_dropout(self.soft_prompt_embeds.expand(batch_size, -1, -1)),
            x[:, 1:, :]
        ), dim=1)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.visual_plm.encoder.eval()
            self.visual_plm.embeddings.eval()
            self.visual_plm.layernorm.eval()
            self.visual_plm.pooler.eval()
            self.soft_prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward(self, input_data):
        embedding_output = self.incorporate_prompt(
            **{k: v for k, v in input_data.items() if k in inspect.signature(self.incorporate_prompt).parameters})
        encoder_outputs = self.visual_plm.encoder(embedding_output)

        last_hidden_state = encoder_outputs['last_hidden_state']
        pooled_output = self.visual_plm.pooler(last_hidden_state)
        soft_hidden_state = self.visual_plm.layernorm(last_hidden_state)[:, 1:1 + self.visual_soft_tokens, :]
        soft_hidden_state = torch.avg_pool1d(soft_hidden_state.transpose(1, 2),
                                             kernel_size=self.visual_soft_tokens).squeeze(-1)
        return {
            'last_hidden_state': soft_hidden_state,
            'pooler_output': pooled_output
        }
class TMPTTextualModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.config = AutoConfig.from_pretrained(args.textual_transformer_name)
        self.hidden_size = self.config.hidden_size
        if "deberta" in args.textual_transformer_name:
            self.textual_plm = AutoModel.from_pretrained(args.textual_transformer_name)
        else:
            self.textual_plm = AutoModel.from_pretrained(args.textual_transformer_name, self.config)

    def extract_at_mask(self, outputs, input_datas):
        if len(outputs.shape) == 2:
            return outputs
        outputs = outputs[torch.where(input_datas['text_loss_ids']>0)]
        outputs = outputs.view(input_datas['text_loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, input_data):
        outputs = self.textual_plm(**{k: v for k, v in input_data.items() if k in inspect.signature(self.textual_plm.forward).parameters})
        outputs_at_mask = {k: self.extract_at_mask(v, input_data) for k, v in outputs.items()}
        return outputs_at_mask
class TMPTModel(nn.Module):
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
        self.stance_classifier = nn.Linear(linear_injection*2, args.label_size)
        self.stance_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_data):
        textual_outputs = self.tmpt_textual_model(input_data)
        text_pooled_output = self.textual_transformer_linear(textual_outputs['last_hidden_state'])

        visual_outputs = self.tmpt_visual_model(input_data)
        image_pooled_output = self.visual_transformer_linear(visual_outputs['last_hidden_state'])

        concat_feat = torch.cat([text_pooled_output, image_pooled_output], dim=-1)

        stance_logits = self.stance_classifier(concat_feat)
        stance_loss = self.stance_loss_fn(stance_logits, input_data['stance_label'])
        return stance_logits, stance_loss
class TMPTModelTextual(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tmpt_textual_model = TMPTTextualModel(args)
        self.stance_loss_fn = nn.CrossEntropyLoss()
        self.stance_classifier = nn.Linear(self.tmpt_textual_model.hidden_size, args.label_size)
    def forward(self, input_data):
        textual_outputs = self.tmpt_textual_model(input_data)
        # text_pooled_output = self.textual_transformer_linear(textual_outputs['last_hidden_state'])
        stance_logits = self.stance_classifier(textual_outputs['last_hidden_state'])
        stance_loss = self.stance_loss_fn(stance_logits, input_data['stance_label'])
        return stance_logits, stance_loss
class TMPTModelVisual(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tmpt_visual_model = TMPTVisualModel(args)
        self.stance_loss_fn = nn.CrossEntropyLoss()
        self.stance_classifier = nn.Linear(self.tmpt_visual_model.hidden_size, args.label_size)
    def forward(self, input_data):
        visual_outputs = self.tmpt_visual_model(input_data)
        # visual_pooled_output = self.visual_transformer_linear(visual_outputs['last_hidden_state'])
        stance_logits = self.stance_classifier(visual_outputs['last_hidden_state'])
        stance_loss = self.stance_loss_fn(stance_logits, input_data['stance_label'])
        return stance_logits, stance_loss
class TMPTModelVisualGate(nn.Module):
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
        self.gate = nn.Sequential(nn.Linear(linear_injection * 2, 2), nn.Sigmoid())
        self.stance_classifier = nn.Linear(linear_injection*2, args.label_size)
    def forward(self, input_data):
        textual_outputs = self.tmpt_textual_model(input_data)
        text_pooled_output = self.textual_transformer_linear(textual_outputs['last_hidden_state'])

        visual_outputs = self.tmpt_visual_model(input_data)
        image_pooled_output = self.visual_transformer_linear(visual_outputs['last_hidden_state'])
        #1)直接拼接
        concat_feat = torch.cat([text_pooled_output, image_pooled_output], dim=-1)
        #2）visual_gate
        # visual_weight = self.visual_gate(image_pooled_output)
        # visual_weighted = image_pooled_output * visual_weight
        # concat_feat = text_pooled_output + visual_weighted
        # # 3) concat_gate
        # gate_value = self.gate(concat_feat)
        # text_gated = text_pooled_output * gate_value
        # image_gated = image_pooled_output * (1 - gate_value)
        # concat_feat = torch.cat([text_gated, image_gated], dim=-1)
        # # 4) concat_gate1
        # gate_value = self.gate(concat_feat)
        # concat_feat = gate_value[:, 0:1] * text_pooled_output+gate_value[:, 1:2] * image_pooled_output
        # 5) concat_gate2
        gate_value = self.gate(concat_feat)
        concat_feat = torch.cat([gate_value[:, 0:1] * text_pooled_output, gate_value[:, 1:2] * image_pooled_output], dim=-1)

        stance_logits = self.stance_classifier(concat_feat)
        stance_loss = self.stance_loss_fn(stance_logits, input_data['stance_label'])
        return stance_logits, stance_loss
class TMPTModelAttention(nn.Module):
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

        # 6)
        # 为每个模态计算Q, K, V
        num_heads=4
        self.num_heads=num_heads
        head_dim = linear_injection*2 // num_heads
        self.q_proj1 = nn.Linear(linear_injection, head_dim * num_heads)
        self.kv_proj1 = nn.Linear(linear_injection, head_dim * num_heads * 2)
        self.q_proj2 = nn.Linear(linear_injection, head_dim * num_heads)
        self.kv_proj2 = nn.Linear(linear_injection, head_dim * num_heads * 2)
        self.output_proj = nn.Linear(head_dim * num_heads * 2, linear_injection)

        self.stance_classifier = nn.Linear(linear_injection*4, args.label_size)

    def forward(self, input_data):
        textual_outputs = self.tmpt_textual_model(input_data)
        text_pooled_output = self.textual_transformer_linear(textual_outputs['last_hidden_state'])

        visual_outputs = self.tmpt_visual_model(input_data)
        image_pooled_output = self.visual_transformer_linear(visual_outputs['last_hidden_state'])
        # 6) attention
        # 计算query, key, value
        batch_size = text_pooled_output.size(0)
        q1 = self.q_proj1(text_pooled_output).view(batch_size, self.num_heads, -1)
        k1, v1 = self.kv_proj1(text_pooled_output).view(batch_size, self.num_heads, 2, -1).unbind(2)
        q2 = self.q_proj2(image_pooled_output).view(batch_size, self.num_heads, -1)
        k2, v2 = self.kv_proj2(image_pooled_output).view(batch_size, self.num_heads, 2, -1).unbind(2)
        # 跨模态注意力：模态1关注模态2
        scores1 = torch.matmul(q1, k2.transpose(-2, -1)) / (q1.size(-1) ** 0.5)
        attn1 = F.softmax(scores1, dim=-1)
        out1 = torch.matmul(attn1, v2)
        # 跨模态注意力：模态2关注模态1
        scores2 = torch.matmul(q2, k1.transpose(-2, -1)) / (q2.size(-1) ** 0.5)
        attn2 = F.softmax(scores2, dim=-1)
        out2 = torch.matmul(attn2, v1)
        # print(out1.shape,out2.shape)
        # 合并输出
        concat_feat = torch.cat([out1, out2], dim=-1).view(batch_size, -1)
        # print(concat_feat.shape)

        stance_logits = self.stance_classifier(concat_feat)
        stance_loss = self.stance_loss_fn(stance_logits, input_data['stance_label'])
        return stance_logits, stance_loss