import logging

import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile
import time
import csv
from collections import defaultdict
import random
from multiprocessing import Process
from paras import transform
import lmdb
from parameters_m2m_nsp import parse_args
args = parse_args()

def _all_videos_lmdb():
    
    all_videos ={}
    ll = [chr(i) for i in range(97, 123)]+ ['300kmiss']
    for a in ll:
        env = lmdb.open('/group/30003/jojiewang/user300k/{}_lmdb_videos'.format(a), map_size=109951162777)#, map_size=1099511627776
        all_videos[a] = env.begin(write=False)
    return all_videos
def _all_videos_lmdb_nc():
    
    all_videos ={}
    ll = [chr(i) for i in range(97, 123)]
    for a in ll:
        env = lmdb.open('/group/30003/jojiewang/articleclient/data/video/{}_lmdb_videos'.format(a), map_size=1099511627776)
        all_videos[a] = env.begin(write=False)
    return all_videos
def smp_collate_fn(data_batch):
    log_mask_batch = torch.stack([item[1] for item in data_batch], dim=0)  # tensor
    mim_mask_batch = torch.stack([item[2] for item in data_batch], dim=0)  # tensor
    label_batch = [item[3] for item in data_batch]
    user_feature_batch = torch.stack([item[0] for item in data_batch], dim=0)  # [tensor]变tensor
    mim_label_batch = torch.stack([item[4] for item in data_batch], dim=0)
    pos_ = torch.stack([item[5] for item in data_batch], dim=0)
    return user_feature_batch, log_mask_batch, mim_mask_batch, label_batch, mim_label_batch, pos_

def smp_collate_fn_projector(data_batch):
    log_mask_batch = torch.stack([item[1] for item in data_batch], dim=0)  # tensor
    label_batch = [item[3] for item in data_batch]
    user_feature_batch = torch.stack([item[0] for item in data_batch], dim=0)  # [tensor]变tensor
    item_feature_batch = torch.stack([item[2] for item in data_batch], dim=0)  # [tensor]变tensor

    return user_feature_batch, log_mask_batch, item_feature_batch, label_batch

def article_collate_fn(arr):
        arr = torch.LongTensor(arr)
        return arr
def video_collate_fn(video):
    video = torch.FloatTensor(video)
    return video

def read_image_loaded(video_id, videos_index, transform):
    video = videos_index[video_id]
    video = Image.fromarray(video)
    if transform is not None:
        video = transform(video)
    video = np.array(video)
    return video

class ArticleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_ids, all_videos, transform=None):
        super(videoDataset, self).__init__()

        self.video_ids = video_ids
        self.transform = transform
        self.all_videos = all_videos


    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        image = self.video_ids[idx]
        #image = read_image(image)
        image = self._read_image_lmdb(image)
        return image
    
    def _read_image_lmdb(self, image):
        name = image
        image = self.all_videos[name[0]].get(name.encode('ascii'))
            #image = self.all_videos[image[0]].get(image, self.all_videos['300kmiss'][image])
        if image==None:
            image = self.all_videos['300kmiss'].get(name.encode('ascii'))
                #print('miss')
        if image==None:
                print('none')
                print(name)
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape([280, 496, 3])
        image = Image.fromarray(image)
        image = transform(image)
        image = np.array(image)

        return image

class VideoDataset_np(torch.utils.data.Dataset):
    def __init__(self, all_videos, transform=None):
        super(videoDataset_np, self).__init__()

        #self.video_ids = video_ids
        self.transform = transform
        self.all_videos = all_videos


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        #image = self.video_ids[idx]
        #image = read_image(image)
        #image = self._read_image_lmdb(image)
        image = self.all_videos[idx]
        return image
    
class TransRecTorchDataset(Dataset):
    def __init__(self, whole_db, whole_index, smp_path=None, transform = None, mode='nsp'):
        super(TransRecTorchDataset,  self).__init__()
        self.whole_db = whole_db
        self.whole_index = whole_index
        self.txt_ids = []    
        self.smp_path = smp_path
        self.all_items = list(self.whole_index)
        #logging.info(self.all_items)
        #logging.info(len(self.all_items)) #48930
        self.data = []
        #self.data = np.load('/group/30003/jojiewang/articleclient/data/qqbrowser_user/o_lines.npy',allow_pickle=True)[500000:520000]   
        lines = np.load('/group/30003/jojiewang/articleclient/data/articleclient/articleclient_mix_seq_filter50k.npy',allow_pickle=True)[:10000]
        self.tiktok = False
        #lines = np.load('/group/30003/jojiewang/articleclient/data/target/tiktok_users.npy',allow_pickle=True)
        #lines = np.load('/group/30003/jojiewang/articleclient/data/articleclient/articleclient_article_seq_filter50k.npy',allow_pickle=True)
        #lines = np.load('/group/30003/jojiewang/articleclient/data/articleclient/articleclient_video_seq_filter50k.npy',allow_pickle=True)
        self.clicks_ = []
        for line in lines:
            if len(line)<3:#article,video,mix, tiktok也用<3测试了
                continue
            self.data.append(line)       
        self.user_log_length = 25
        self.npratio = 2
        self.transform = transform
        self.enable_gpu = True
        self.neg_sample = np.load('/group/30003/jojiewang/articleclient/data/neg_sample_4k.npy')
        self.mode = mode
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        line = self.data[idx]

        user_feature, log_mask, mim_mask, label, mim_label,pos_ids = self._process(line, self.neg_sample[0])

        return user_feature, log_mask, mim_mask, label, mim_label,pos_ids
    def trans_to_nindex(self, nids, index_db):

        return [index_db[i] for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=False, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value]*(fix_length-len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, mask

    def select_items(self, items, type = 'click'):
        pos_ids = []
        for i, k in enumerate(items):
            if k.startswith('2'):
                pos_ids.append(1)
            else:
                pos_ids.append(0)
        if type == 'click' and len(pos_ids) < self.user_log_length:
            pos_ids.extend([-1]*(self.user_log_length-len(pos_ids)))
        elif type == 'smp':
            pos_ids.extend([-1] * (50 - len(pos_ids)))

        return pos_ids

    def _process(self, line, neg):
        user_feature, item_feature = [], []
        if self.mode=='nsp' or self.mode=='nip':
            clicks = line[-(self.user_log_length+1):-1]
        elif self.mode=='ft_msk' or self.mode=='msk':
            clicks = line[-self.user_log_length:]
        if self.tiktok:
            pos_ids = [0]*len(clicks)+[-1]*(self.user_log_length-len(clicks))
        else:
            pos_ids = self.select_items(clicks, 'click')
        pos_item = line[-1:]
        user_feature = self._inputitem(clicks)
        if len(clicks) <  self.user_log_length:
            user_feature.extend(np.zeros((self.user_log_length - len(clicks), args.item_dim), dtype=np.int))            
        #log_mask = [1]*(len(clicks)+1) + [0]*(self.user_log_length - len(clicks))
        if self.mode=='ft_msk':
            log_mask = [1]*(len(clicks)+1) + [0]*(self.user_log_length - len(clicks))
        elif self.mode=='msk' or self.mode=='nsp':
            log_mask = [1]*(len(clicks)) + [0]*(self.user_log_length - len(clicks))
        elif self.mode=='nip':
            log_mask = [1]*(len(clicks)+1) + [0]*(self.user_log_length - len(clicks))
        mim_mask = [0]*(len(clicks)-1) + [1] + [0]*(self.user_log_length - len(clicks))

        if self.smp_path=='/group/30003/jojiewang/articleclient/data/articleclient_mix_seq':
            #neg_items = random.sample(list(set(self.all_items).difference(set(line))), 100)
            neg_items = neg.tolist()
        elif self.smp_path=='/group/30003/jojiewang/articleclient/data/articleclient_article_seq':
            neg_items = random.sample(list(set(self.txt_ids).difference(set(line))), 100)
        elif self.smp_path=='/group/30003/jojiewang/articleclient/data/articleclient_video_seq':
            neg_items = random.sample(list(set(self.video_index.keys()).difference(set(line))), 100)
        #item_feature = self._inputitem(pos_item + neg_items)
        mim_label = np.array([0] * len(self.whole_index), dtype=float)
        if self.tiktok:
            labels = line
        else:
            labels = self.trans_to_nindex(line, self.whole_index)
        mim_label[labels[:-1]] = -10000 
        #mim_label[0] = -10000 
        label = labels[-1]
        user_feature = torch.FloatTensor(user_feature)#更新了传入的db为tensor
        #item_feature = torch.FloatTensor(item_feature)
        log_mask = torch.LongTensor(log_mask)
        mim_mask = torch.LongTensor(mim_mask)
        pos_ids = torch.LongTensor(pos_ids)
        mim_label = torch.LongTensor(mim_label)

        label = np.array(label)
        return user_feature, log_mask, mim_mask, label, mim_label,pos_ids
    
    def _inputitem(self, items):
        feature = []
        if not self.tiktok:
            clk = self.trans_to_nindex(items, self.whole_index)
            feature.extend(self.whole_db[clk])  
        else: 
            feature.extend(self.whole_db[items])  
        return feature