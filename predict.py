import torch
import os
import logging
import sys
sys.path.append('./')
from data_utils.tools import set_seed, load_image, read_yaml, AttributeDict,load_csv
from data_utils.social_media_texts_utils import clean_text
from transformers import AutoTokenizer, AutoImageProcessor

# 基础配置
logging.basicConfig(level=logging.INFO)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from data_utils.data_utils_tmpt import TMPTDataset as datasets
from data_utils.data_config import data_configs as data_configs
from openprompt.data_utils import InputExample
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.prompts import ManualTemplate

class TMPTInference:
    def __init__(self, args):
        self.args=args
        self.dataset_name = args.dataset_name
        self.framework_name = args.framework_name
        self.model_name = args.model_name

        # 1. 加载数据集配置（获取label映射、target文本模板等）
        self.data_config = data_configs[self.dataset_name]
        self.label2idx = self.data_config.label2idx
        self.idx2label = {v: k for k, v in self.label2idx.items()}

        self.textual_tokenizer = AutoTokenizer.from_pretrained(args.textual_transformer_tokenizer_name,
                                                          model_max_length=args.max_tokenization_length,
                                                          use_fast=False)
        self.visual_processor = AutoImageProcessor.from_pretrained(args.visual_transformer_tokenizer_name)

        # 4. 加载预训练模型
        short_target = self.data_config.short_target_names[test_target]
        if args.in_target:
            OUTPUT_DIR = os.path.dirname(__file__) + '/outputs' + (f'/{self.dataset_name}_in_target_'
                        f'{self.framework_name}_{self.model_name}')#_expert_4_topk_2
        else:
            OUTPUT_DIR = os.path.dirname(
                __file__) + '/outputs' + f'/{self.dataset_name}_zero_shot_{self.framework_name}_{self.model_name}'
        file_prefix=f'{self.dataset_name}_in_target_{short_target}_seed_{args.seed}'
        print(file_prefix)
        print(os.listdir(OUTPUT_DIR+ f'/models'))
        # matched_files = [f for f in os.listdir(OUTPUT_DIR+ f'/models') if f.startswith(file_prefix)]
        matched_files = [
            f for f in os.listdir(OUTPUT_DIR+ f'/models')
            if f.startswith(file_prefix)  # 保留原有开头匹配条件
               and (
                   # 条件1：文件恰好等于file_prefix（边界情况，一般模型文件不会出现，可选保留）
                       len(f) == len(file_prefix)
                       or (
                           # 条件2：file_prefix之后的第一个字符不是数字
                               not f[len(file_prefix)].isdigit()
                               and (
                                   # 条件3：要么以.pt结尾（模型权重文件）
                                       f.endswith(".pt")
                                       # 条件4：要么file_prefix之后紧跟下划线
                                       or (len(f) > len(file_prefix) and f[len(file_prefix)] == "_")
                               )
                       )
               )
        ]

        first_file = matched_files[0]
        model_path = OUTPUT_DIR + f'/models/'+ first_file
        logging.info(f'加载模型权重：{model_path}')
        if args.framework_name == "moe_miloss_visualgate_tmpt":
            from models.tmpt_moe1 import TMPTModelVisualGateMOEMiloss1 as Model
        elif args.framework_name == 'tmpt':
            from models.tmpt import TMPTModel as Model
        self.model = Model(config).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        logging.info('Model loaded successfully!')
        self.model.eval()  # 切换到评估模式，禁用梯度
    def tmpt_text_prompting_encode(self, sentence, target):
        template_text = '{"placeholder":"text_a", "shortenable": True}. {"placeholder":"text_b", "shortenable": False} {"mask"}.'
        # shortenable截断
        prompting_template = ManualTemplate(tokenizer=self.textual_tokenizer, text=template_text)
        wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=self.textual_tokenizer.model_max_length,
                                                tokenizer=self.textual_tokenizer, truncate_method="tail")

        prompting_data = InputExample(text_a=sentence, text_b=target)
        encoding = wrapped_tokenizer.tokenize_one_example(prompting_template.wrap_one_example(prompting_data),
                                                          teacher_forcing=False)
        return encoding

    def process_input(self, target, text, image_path):
        input_ids = []
        loss_ids = []
        attention_mask = []
        # 1. 加载并处理图片
        image_path=f"{self.data_config.data_dir}/{image_path}"
        image = load_image(image_path)  # 复用原项目的图片加载逻辑
        image_encoding = self.visual_processor(
            images=image,
            return_tensors='pt'
        )
        target=self.data_config.topic_text[target]
        encoding=self.tmpt_text_prompting_encode(text,target)
        input_ids.append(encoding['input_ids'])
        loss_ids.append(encoding['loss_ids'])
        attention_mask.append(encoding['attention_mask'])
        encodings={'input_ids': torch.tensor(input_ids), 'text_loss_ids': torch.tensor(loss_ids),
         'attention_mask': torch.tensor(attention_mask)}
        encodings.update(image_encoding)
        encodings['stance_label'] = torch.tensor([0])
        return encodings

    @torch.no_grad()  # 禁用梯度，提升速度并节省显存
    def predict(self, target, text, image_path):
        # 1. 处理输入
        model_input = self.process_input(target, text, image_path)
        model_input = {k: v.to(DEVICE) for k, v in model_input.items()}
        # 2. 模型推理
        if self.args.framework_name == "moe_miloss_visualgate_tmpt":
            logits, _ = self.model(model_input, top_k=self.args.top_k)
        elif self.args.framework_name == 'tmpt':
            logits, _ = self.model(model_input)
        # 3. 解析结果
        pred_idx = torch.argmax(logits, dim=-1).item()
        pred_label = self.idx2label[pred_idx]
        pred_confidence = torch.softmax(logits, dim=-1)[0][pred_idx].item()

        # 4. 返回结构化结果
        return {
            'pred_label': pred_label,  # 预测的立场标签（如support/oppose/neutral）
            'pred_confidence': round(pred_confidence, 4),  # 预测置信度
        }


# -------------------------- 测试示例 --------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default="imgarg")
    parser.add_argument('--framework_name', type=str, default='moe_miloss_visualgate_tmpt')  # base visual textual multimodal
    parser.add_argument('--model_name', type=str, default='deberta_vit')  # bert_base resnet multimodal
    parser.add_argument('--cot', action='store_true', default=False)
    parser.add_argument('--in_target', action='store_true',default=True)
    parser.add_argument('--zero_shot', action='store_true',default=False)
    parser.add_argument('--normal', action='store_true',default=True)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    GLOBAL_ARGS = parser.parse_args()

    raw_config = read_yaml('config/config.yaml')
    if GLOBAL_ARGS.model_name in raw_config['model_config']:
        raw_config['model_config'] = raw_config['model_config'][GLOBAL_ARGS.model_name]
    else:
        raise ValueError('Invalid model')
    raw_config['model_config']['label_size']['value'] = len(data_configs[GLOBAL_ARGS.dataset_name].label2idx)

    if GLOBAL_ARGS.in_target:
        raw_config['train_config'] = raw_config['train_config']['in_target_train_config']
    elif GLOBAL_ARGS.zero_shot:
        raw_config['train_config'] = raw_config['train_config']['zero_shot_train_config']
    config = {}
    for value in raw_config.values():
        for k, v in value.items():
            if 'value' in v:
                config[k] = v['value']
            elif 'values' in v:
                config[k] = v['values'][0]
    args_dict = vars(GLOBAL_ARGS)
    config = {**config, **args_dict}## 合并字典，命令行参数优先（覆盖已有字典）
    print(config)
    config = AttributeDict(config)

    config.seed= 42#2022
    set_seed(config.seed)
    # 输入：目标+文本+图片（替换为你的实际测试数据）
    test_target = "abortion"#Joe Biden"#Donald Trump abortion"gun control"  # 目标话题
    test_text = "Is Abortion in Israel Holding Back Moshiach? The 40,000 a year abortion free-for-all in Israel is horrifying, shameful, and  devastating. Please contact us if you can volunteer to help us bring our pro-life education to Israel. https://t.co/8N93Yz08mC https://t.co/SMaoTM33av"
    test_image_path = "images/abortion/1343172775696084992.jpg"  # 图片路径
    predictor = TMPTInference(args=config)
    # 执行预测
    result = predictor.predict(test_target, test_text, test_image_path)
    # 打印结果
    logging.info("=== 预测结果 ===")
    for k, v in result.items():
        logging.info(f"{k}: {v}")