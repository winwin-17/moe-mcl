
class MTSEConfig():
    dataset_name = 'Multi-modal Twitter Stance Election 2020'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'Favor': 0, 'Against': 1, 'Neutral': 2}
    target_names = ['Donald Trump', 'Joe Biden']
    zero_shot_target_names = ['Donald Trump','Joe Biden']
    short_target_names = {
        'Donald Trump': 'DT',
        'Joe Biden': 'JB',
    }
    topic_text = {
        'Donald Trump': 'The stance on Donald Trump is:',
        'Joe Biden': 'The stance on Joe Biden is:'
    }

    data_dir = 'dataset/Multi-Modal-Stance-Detection/Multi-modal-Twitter-Stance-Election-2020'
    data_dir = '/home/xxx/stance_dataset/Multi-Modal-Stance-Detection/2020-US-Presidential-Election'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'

class ImgArgConfig():
    dataset_name = 'ImgArg'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1]
    label2idx = {'support': 0, 'oppose': 1}#, 'neutral': 2,'none':2}
    target_names = ['abortion','gun control']
    zero_shot_target_names = ['abortion','gun control']

    short_target_names = {
        'gun control': 'GC',
        'abortion': 'AB',
    }
    topic_text = {
        'gun control': 'The stance on gun control is:',
        'abortion': 'The stance on abortion is:'
    }

    data_dir = 'dataset/Multi-Modal-Stance-Detection/Multi-modal-Twitter-Stance-Election-2020'
    data_dir='/home/xxx/stance_dataset/Multi-Modal-Stance-Detection/ImgArg'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot1'

data_configs = {
    'mtse': MTSEConfig,
    "imgarg":ImgArgConfig,
}
