from __future__ import absolute_import  # 在当前版本引入新版本的特性
from __future__ import division
from __future__ import print_function

# 以上为开源项目经常使用的语句，目的是为了增加代码的普适性，分别为绝对import，除法，打印

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()

# Input paths
parser.add_argument('--model', type=str, default='', help='path to model to evaluate')  # 模型地址
parser.add_argument('--cnn_model', type=str, default='resnet101', help='resnet101, resnet152')  # 预训练地址
parser.add_argument('--infos_path', type=str, default='', help='path to infos to evaluate')  # infos地址
opts.add_eval_options(parser)  # 将新的输入参数加载到opt中

opt = parser.parse_args()  # 调用parse_args()进行解析，返回值为namespace，所有参数以属性的方式存在

# Load infos
with open(opt.infos_path, 'rb') as f:  # 读取infos文件地址，返回文件对象，使用utils.pickle_load()序列化读出
    infos = utils.pickle_load(f)  # pickle可以理解为一个io工具，负责将文件序列化与反序列化的存储与读取

# override and collect parameters 重新对参数进行设置，以适应eval
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():  # vars是返回对象（字典）的属性与键值的函数
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))  # setattr设置对象属性值，getattr获取对象属性值
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
opt.vocab = vocab  # 将infos的语料库赋值到opt的参数vocab
model = models.setup(opt)  # 将模型进行按照opt的参数进行初始化
del opt.vocab
model.load_state_dict(torch.load(opt.model))  # 将模型加载到torch中
model.cuda()  # 加载cuda
model.eval()  # 不启用 BatchNormalization 和 Dropout
crit = utils.LanguageModelCriterion()  # 将LanguageModelCriterion实例化

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,
                                                            vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
