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
import io
import lmdb

def _all_videos_lmdb():
    
    all_videos ={}
    ll = [chr(i) for i in range(97, 123)]
    for a in ll:
        env = lmdb.open('/video/{}_lmdb_videos'.format(a), map_size=1099511627776)
        all_videos[a] = env.begin(write=False)
    return all_videos            
    

def smp_collate_fn(data_batch):
    user_feature_batch, item_feature_batch = defaultdict(list), defaultdict(list)
    log_mask_batch = torch.stack([item[1] for item in data_batch], dim=0)
    log_mask1_batch = torch.stack([item[3] for item in data_batch], dim=0)

    user_feature_batch['pos_ids'] = torch.stack([item[0]['pos_ids'] for item in data_batch], dim=0)
    user_feature_batch['txt_item'] = torch.cat([item[0]['txt_item'] for item in data_batch if type(item[0]['txt_item'])==torch.Tensor], dim=0)
    #user_feature_batch['txt_item']  = torch.stack([item[0]['txt_item'] for item in data_batch], dim=0)
    user_feature_batch['videos_item'] = torch.cat([item[0]['videos_item'] for item in data_batch if type(item[0]['videos_item'])==torch.Tensor], dim=0) 
    #user_feature_batch['videos_item']  = torch.FloatTensor([item[0]['videos_item'] for item in data_batch])
    
    item_feature_batch['pos_ids'] = torch.stack([item[2]['pos_ids'] for item in data_batch], dim=0)
    #item_feature_batch['txt_item'] = torch.stack([item[2]['txt_item'] for item in data_batch], dim=0)
    item_feature_batch['txt_item'] = torch.cat([item[2]['txt_item'] for item in data_batch if type(item[2]['txt_item'])==torch.Tensor], dim=0)
    #item_feature_batch['videos_item'] = torch.FloatTensor([item[2]['videos_item'] for item in data_batch])
    item_feature_batch['videos_item'] = torch.cat([item[2]['videos_item'] for item in data_batch if type(item[2]['videos_item'])==torch.Tensor], dim=0)
    return user_feature_batch, log_mask_batch, item_feature_batch, log_mask1_batch

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
def trans_to_nindex_(nids):

        rr_index = {}
        index = 0
        for i in nids:
            rr_index[i] = index
            index+=1
        return rr_index 
        
class TransRecTorchDataset(Dataset):
    def __init__(self, txt_db, txt_index, all_videos, video_index, _clicks, o_line_ , transform = None):
        super(MFlowTorchDataset,  self).__init__()
        self.txt_db = txt_db
        self.video_index = video_index
        self.txt_index = txt_index
        ids = np.load('xxx.npy', allow_pickle=True)
        self.all_items = ids
        self.data = []
        self.user_log_length = 25
        self.npratio = 5
        self.ppratio = 5

        self.transform = transform
        self.enable_gpu = True
        self.users, self.items, self.labels = [], [], []
        self.clicks_ = _clicks
        self.o_line_ = o_line_
        self.all_videos = all_videos
 
    def __len__(self):
        return len(self.clicks_)

    def __getitem__(self, idx):
        user, item, log_mask, mim_mask = self._sample_negative(self.clicks_[idx], self.npratio, self.o_line_[idx])
        user_feature, item_feature = self._process(user, item)
        log_mask = torch.LongTensor(log_mask)
        mim_mask = torch.LongTensor(mim_mask)
        return user_feature, log_mask, item_feature, mim_mask
       
    def _read_image(self, image):
        name = image
        try:
            image = self.all_videos[name[0]][name]
        except Exception as e:
            try:
                image = self.all_videos['300kmiss'][name]
                #print('miss')
            except Exception as e:
                print('none')
                print(name)
        image = image.convert('RGB')
        image = transform(image)
        image = np.array(image)

        return image
    
    def _read_image_lmdb(self, image):
        name = image
        image = self.all_videos[name[0]].get(name.encode('ascii'))
        if image==None:
            image = self.all_videos['300kmiss'].get(name.encode('ascii'))
        if image==None:
                print('none')
                print(name)
        image = np.frombuffer(image, dtype=np.uint8)
        image = image.reshape([280, 496, 3])
        image = Image.fromarray(image)
        image = transform(image)
        image = np.array(image)

        return image
    def trans_to_nindex(self, nids, nindex):

        return [nindex[i] for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=False, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = np.array(x[:fix_length] + [padding_value]*(fix_length-len(x)))
            mask = np.array([1] * min(fix_length, len(x)) + [0] * (fix_length - len(x)))
        return pad_x, mask

    def select_items(self, items, type = 'click'):
        txt_items, videos_items = [], []
        pos_ids = []
        for i, k in enumerate(items):
            if k.startswith('2'):
                txt_items.append(k)
                pos_ids.append(1)
            else:
                videos_items.append(k)
                pos_ids.append(0)
        if type == 'click' and len(pos_ids) < self.user_log_length:
            pos_ids.extend([-1]*(self.user_log_length-len(pos_ids)))

        return txt_items, videos_items, pos_ids
    
    def random_item(self, log_mask):
        #mask_feats = feats.copy()
        mim_mask = log_mask.copy()
        for i in range(sum(log_mask)):
            prob = random.random()
            # mask token with probability
            #if prob < args.obj_mask_rate:
            #logging.info(prob)
            if prob < 0.15: #629eps,0.3，630后，0.15
                mim_mask[i] = 0
        return mim_mask

    def _sample_negative(self, clicks, num_negatives, o_line):

        click_txt_items, click_videos_items, click_pos_ids = self.select_items(clicks[-(self.user_log_length+self.ppratio):-self.ppratio], 'click')
        click_txt_items = self.trans_to_nindex(click_txt_items, self.txt_index)
        click_videos_items = self.trans_to_nindex(click_videos_items, self.video_index)
        num =len(click_txt_items) + len(click_videos_items)
        log_mask = [1]* num + [0]*(self.user_log_length-num)
        mim_mask = [0]* (num-1) + [1]*1 + [0]*(self.user_log_length-num)

        neg_item = random.sample(list(set(self.all_items).difference(set(o_line))), self.npratio)
        neg_txt_items, neg_videos_items, neg_pos_ids = self.select_items(clicks[-self.ppratio:] + neg_item, 'smp')
        neg_txt_items = self.trans_to_nindex(neg_txt_items, self.txt_index)
        neg_videos_items = self.trans_to_nindex(neg_videos_items, self.video_index)
        
        return [click_txt_items, click_videos_items, click_pos_ids], [neg_txt_items, neg_videos_items, neg_pos_ids],log_mask, mim_mask

    def _inputdict(self, user):
        user_feature = defaultdict(list)
        click_txt_items, click_videos_items, click_pos_ids = user#txt index,video id,pos
        user_feature['pos_ids'] = click_pos_ids
        if click_txt_items:
            user_feature['txt_item'] = self.txt_db[click_txt_items]
        if click_videos_items:
            #user_feature['videos_item'] = list(map(self._read_image_lmdb,click_videos_items))
            user_feature['videos_item'] = self.all_videos[click_videos_items]


        return user_feature

    def _smp_inputdict(self, item):
        item_feature = defaultdict(list)
        neg_txt_items, neg_videos_items, neg_pos_ids = item
        item_feature['pos_ids'] = neg_pos_ids
        if neg_txt_items:
            item_feature['txt_item'] = self.txt_db[neg_txt_items]
        if neg_videos_items:
            #item_feature['videos_item'] = list(map(self._read_image_lmdb,neg_videos_items))
            item_feature['videos_item'] = self.all_videos[neg_videos_items]
        return item_feature


    def _process(self, user, item):
        user_feature = self._inputdict(user)

        user_feature['pos_ids'] = torch.LongTensor(user_feature['pos_ids'])
        #user_feature['txt_item'] = torch.LongTensor(user_feature['txt_item'])
        #user_feature['videos_item'] = torch.FloatTensor(user_feature['videos_item'])
        item_feature = self._smp_inputdict(item)
        item_feature['pos_ids'] = torch.LongTensor(item_feature['pos_ids'])
        #item_feature['txt_item'] = torch.LongTensor(item_feature['txt_item'])
        #item_feature['videos_item'] = torch.FloatTensor(item_feature['videos_item'])
        return user_feature, item_feature
