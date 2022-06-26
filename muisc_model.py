
import logging
import math
import os
import sys
from typing import Optional
from collections import OrderedDict
import numpy as np

from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    BertLayer,
    GPT2DoubleHeadsModel,
    ViTModel,
    ViTFeatureExtractor,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import argparse

from PIL import Image


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



class DS(Dataset):

    def _read_image_data_from_file(self, filenames, shape=(224, 224, 3)):
        try:
            image = [Image.open(filename).convert('RGB') for filename in filenames]
            image_data = self.feature_extractor(images=image, return_tensors="pt") #['pixel_values']
            #print('_read_image_data_from_file: Got one. {}'.format(image_data.shape))
        except:
            image_data = self.feature_extractor(images=[np.zeros(shape)]*3, return_tensors="pt")  # hard-encoded
            print("An exception occurred in _read_image_data_from_file {}.".format(filenames))

        return image_data


    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tok = tokenizer
        self.max_length = max_length

        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        assert 'img_0' in self.data[index] and 'img_1' in self.data[index] and 'img_2' in self.data[index]
        img_files = [self.data[index]['img_0'], self.data[index]['img_1'], self.data[index]['img_2']]

        #
        if 'cid1' in self.data[index]:
            cid1 = self.data[index]['cid1']
        else: cid1 = 0
        if 'cid2' in self.data[index]:
            cid2 = self.data[index]['cid2']
        else: cid2 = 0
        cid_text = "_".join([str(cid1), str(cid2)])

        if 'title' in self.data[index]:
            title = self.data[index]['title']
        else:
            title = ''

        #
        mc_labels = 0
        offreason = self.data[index]['offreason']
        #text = cid_text + "_" + title + "[MASK]" + offreason
        text = title + "[MASK]" + offreason
        print("text = {}".format(text))
       

        line = self.tok.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        line = {kk: vv.squeeze(0) for kk, vv in line.items()}
    
        imgs = self._read_image_data_from_file(img_files)
        line['image_data'] = imgs
        line['image_files'] = img_files

        line['mc_token_ids'] = line['input_ids'].tolist().index(self.tok.mask_token_id) 
        line['labels'] = line['input_ids'].clone()
        for ii in range(line['mc_token_ids']+0): # including mc_toke itself
            line['labels'][ii] = -100

        line['offreason'] = offreason
        line['mc_labels'] = mc_labels
        line['wid'] = 0

        return line
    

class ImageEncoderModel(nn.Module):
    def __init__(self, out_embed, pos_num=197*3):
        super(ImageEncoderModel, self).__init__()

        single_image_model_name = 'google/vit-base-patch16-224-in21k'
        single_image_config = AutoConfig.from_pretrained(single_image_model_name)
        #single_image_config.num_hidden_layers = 3
        self.image_tower = ViTModel.from_pretrained(single_image_model_name, config=single_image_config)
        print('xc====== single_image_config = {}'.format(single_image_config))


        self.use_image_cross = True
        self.pos_num = pos_num

        if self.use_image_cross is True:
            print("xc====== Use BertLayer")
            cross_image_config = AutoConfig.from_pretrained('bert-base-chinese')
            cross_image_config.num_hidden_layers = 1
            # cross_image_config.num_attention_heads = 12
            cross_image_config.add_cross_attention = False
            cross_image_config.is_decoder = False

            self.self = BertLayer(cross_image_config)
            self.wpe = nn.Embedding(pos_num, out_embed)
            self.position_ids = torch.tensor(list(range(pos_num)), dtype=torch.int32)

            #"""Initialize weights"""
            #self.wpe.apply(self._init_weights)

        #if  self.use_image_cross == True:
            #self.self.apply(self._init_weights)

        print('xc====== use_image_cross is {}'.format(self.use_image_cross))

        #self.encoder_attention_masks = self._build_encoder_attention_mask(image_seq_len=pos_num, head_num=12) # pos_num
        self.encoder_attention_masks = None
        self.dropout_prob = 0.0
        print("xc====== ImageEncoderModel: dropout_prob = {}".format(self.dropout_prob))


    def forward(self, image_data=None, device=None, eval=True):
        
        if image_data is None:
            return None
        else:
            input = image_data
            if device is not None:
                input['pixel_values'] = input['pixel_values'].to(device)


            assert len(input['pixel_values'].size()) == 4 or len(input['pixel_values'].size()) == 5
            if len(input['pixel_values'].size()) == 4:
                batch_size = 1
            else:
                batch_size = input['pixel_values'].shape[0]
                input['pixel_values'] = input['pixel_values'].view((-1, input['pixel_values'].shape[-3],
                                                                    input['pixel_values'].shape[-2],
                                                                    input['pixel_values'].shape[-1]))


            outputs = self.image_tower(**input)
            x = outputs.last_hidden_state

            if self.use_image_cross == True:
                position_embeds = self.wpe(self.position_ids.to(input['pixel_values'].device))
                xx = x.view((batch_size, -1, x.shape[-1]))
                x = self.self(xx + position_embeds,  attention_mask=None)[0]

            x = x.view((batch_size, 1, -1, x.shape[-1]))

            return x





