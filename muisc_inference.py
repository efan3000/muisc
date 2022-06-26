import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, TextGenerationPipeline, GPT2DoubleHeadsModel
from muisc_model import ImageEncoderModel, DS
#from training_paper_for_lang_v0_allcls_multitask  import ImageEncoderModel, DS
import datasets
from datasets import load_dataset
import os
from torch.utils.data import Dataset, DataLoader
import json


def load_model(model_filename):

    lang_model_name = "uer/gpt2-chinese-cluecorpussmall"


    print('Load model from {}'.format(model_filename))
    lang_state_dict = {
        key[11:]: value
        for key, value in torch.load(
        model_filename,
        map_location="cpu").items() if 'lang_model.' in key
    }

    config = AutoConfig.from_pretrained(lang_model_name)
    config.n_layer = 3
    config.add_cross_attention = True
    #
    config.summary_use_proj = True
    config.summary_type = 'cls_index'
    config.num_labels = total_cls_num
    # config.summary_proj_to_labels = True
    # print('config.summary_first_dropout = {}'.format(config.summary_first_dropout))
    tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    # lang_model = GPT2LMHeadModel(config=config)
    lang_model = GPT2DoubleHeadsModel(config=config)
    lang_model.load_state_dict(lang_state_dict)
    print('lang_model loaded')
    lang_model.eval()
    lang_model.to(device)

    img_state_dict = {
        key[10:]: value
        for key, value in torch.load(
        model_filename,
        map_location="cpu").items() if 'img_model.' in key
    }

    img_config = AutoConfig.from_pretrained('bert-base-chinese')
    img_config.num_hidden_layers = 1
    # img_config.num_attention_heads = 12
    img_config.add_cross_attention = False
    img_config.is_decoder = False
    img_model = ImageEncoderModel(out_embed=config.n_embd)
    img_model.load_state_dict(img_state_dict, strict=True)
    img_model.eval()
    img_model.to(device)
    print('img_model loaded')

    return lang_model, img_model, tokenizer

def build_data():

    def write_lines_to_json_file(lines, out_filename):
        print('Write json dataset to {}'.format(out_filename))
        with open(out_filename, "w") as outF:
            for ii in range(len(lines)):
                outF.write(lines[ii])
                outF.write('\n')

    temp_dataset = []
    img_0 = image_path[0]
    img_1 = image_path[1]
    img_2 = image_path[2]

    one_data = {'img_0': img_0,
                'img_1': img_1,
                'img_2': img_2, 'offreason': 'yes'}
    one_data['cid1'] = cid1
    one_data['cid2'] = cid2
    one_data['title'] = title
    one_data['wid'] = wid
    one_line = json.dumps(one_data, ensure_ascii=False).encode('utf8').decode()
    temp_dataset.append(one_line)

    #
    test_data_filename = 'temp_cushai_datsaset_{}.json'.format(len(temp_dataset))
    write_lines_to_json_file(temp_dataset, test_data_filename)
    data_files = {}
    data_files["validation"] = test_data_filename
    extension = test_data_filename.split(".")[-1]
    data = load_dataset(
        extension,
        data_files=data_files,
        # split=
    )
    for kk, vv in data.items():
        print('valid_data has {}'.format(kk))

    raw_valid_data = data['validation']
    print('raw_valid_data = {} {}'.format(raw_valid_data.shape, raw_valid_data[0]))
    valid_data = DS(raw_valid_data, tokenizer, max_length=max_length)

    return valid_data


def inference():

    m = nn.Softmax(dim=1)
    batch_size = 1
    loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=True,
    )

    for batch_idx, sample in enumerate(loader):
        image_data = sample['image_data']
        image_feat = img_model(image_data, device=device, eval=True)
        mc_token_ids = sample['mc_token_ids']
        mc_label_tensor = sample['mc_labels']
        mc_labels = mc_label_tensor.cpu().detach().tolist()

        lang_model_input = {'input_ids': sample['input_ids'].to(device),
                            'attention_mask': sample['attention_mask'].to(device),
                            'labels': sample['labels'].to(device),
                            'return_dict': True,
                            'encoder_hidden_states': image_feat.to(device),
                            'mc_labels': mc_label_tensor.to(device),
                            'encoder_attention_mask': None,
                            'mc_token_ids': mc_token_ids.to(device),
                            'cross_ids': cross_ids,
                            }
        r = lang_model(**lang_model_input)

        cls_probs = m(r['mc_logits'][:, :valid_cls_num]).cpu().detach().numpy()
        print("cls_probs = {}".format(cls_probs))



if __name__ == "__main__":

    model_filename = "./model/muisc_model_001.ckpt"    

    image_path = ["./example_images_012/001.jpg", "./example_images_012/004.jpg", "./example_images_012/003.jpg"]
    title = 'adidas阿迪达斯官网ADILETTE SHOWER 高桥理子联名女子游泳运动凉鞋拖鞋FX1200 黑/白 38(235mm)'
    cid1 = 0
    cid2 = 0
    wid = 0


    total_cls_num = 45
    valid_cls_num = 45
    max_length = 128
    cross_ids = [2]

    cuda_ = "cuda"
    device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print('device = {}'.format(device))

    """Load model"""
    lang_model, img_model, tokenizer = load_model(model_filename)

    """Build data"""
    test_data = build_data()

    inference()


