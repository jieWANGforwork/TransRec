import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import numpy as np
import pickle
import hashlib
import logging
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import sys
import logging
from torch.utils.data import Dataset, DataLoader
from preprocess import read_article_bert, get_doc_input_bert
from model import TransRec
from resnet import resnet18
from bert import BertModel
from parameters import parse_args
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datetime import datetime
from data_test import TransRecTorchDataset, smp_collate_fn, ArticleDataset, article_collate_fn, VideoDataset, VideoDataset_np, video_collate_fn,  _all_videos_lmdb, _all_videos_lmdb_nc
import torchvision
from paras import transform

def test(args, sample, model_path, ckpt, step):
    if True:
        import horovod.torch as hvd
        hvd.init()
        logging.info(
        f"hvd_size:{hvd.size()}, hvd_rank:{hvd.rank()}, hvd_local_rank:{hvd.local_rank()}"
            )

        hvd_size = hvd.size() if args.enable_hvd else 1
        hvd_rank = hvd.rank() if args.enable_hvd else 0
        hvd_local_rank = hvd.local_rank() if args.enable_hvd else 0

        if args.enable_gpu and torch.cuda.is_available():
            torch.cuda.set_device(hvd_local_rank)
    print(hvd_size, hvd_rank, hvd_local_rank)

    if ckpt in ['0','-1','-2']:
        if ckpt=='0':
            ckpt_path = 'xxx'
        if ckpt=='-1':
            ckpt_path = 'xxx'
        if ckpt=='-2':
            ckpt_path = 'xxx'
    else:
        if args.load_ckpt_name is not None:
            ckpt_path = utils.get_checkpoint(model_path, 'epoch-{}-{}.pt'.format(ckpt, step))
        else:
            ckpt_path = utils.latest_checkpoint(model_path)

    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path)
    logging.info('loaded params')

    tokenizer = AutoTokenizer.from_pretrained('/workspace/user_code/MFlow/bert/bert-base-chinese')
    config = AutoConfig.from_pretrained('/workspace/user_code/MFlow/bert/bert-base-chinese', output_hidden_states=True)
    bert_model = BertModel(config=config)
    resnet_model = resnet18(pretrained=False)
    model = TransRec(args, bert_model, resnet_model)

    if args.enable_gpu:
        model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    article, article_index = read_article_bert('xxx', args, tokenizer)

    article_title, article_title_type, article_title_attmask, \
    article_abstract, article_abstract_type, article_abstract_attmask, \
    article_body, article_body_type, article_body_attmask= get_doc_input_bert(
        article, article_index, args)
    txt_item_doc = np.concatenate([article_title, article_title_type, article_title_attmask], axis=1)  # 按列拼接，列变多
    logging.info('article loaded')
    article_scoring = []
    if step!='647':
        article_dataset = articleDataset(txt_item_doc)  
        article_dataloader = DataLoader(article_dataset,
                                     batch_size=1024,
                                     shuffle=False,
                                     num_workers=1,
                                     collate_fn=article_collate_fn)

        from collections import defaultdict
        with torch.no_grad():
            for input_ids in tqdm(article_dataloader):
                # logging.info(input_ids.shape)
                input_items = defaultdict(list)
                input_items['vis_item'] = torch.FloatTensor(input_items['vis_item']).cuda()
                pos_ids = [1] * input_ids.shape[0]
                input_items['txt_item'].extend(input_ids)
                input_items['txt_item'] = torch.LongTensor(
                    [item.cpu().detach().numpy() for item in input_items['txt_item']]).cuda()
                input_items['pos_ids'].append(pos_ids)
                input_items['pos_ids'] = torch.LongTensor(input_items['pos_ids']).cuda()
                # input_items['txt_item'] = torch.LongTensor(input_items['txt_item']).cuda()
                article_vec = model.item_encoder(input_items)
                article_vec = article_vec.to(torch.device("cpu")).detach().numpy()
                article_scoring.extend(article_vec)  # n, dim[[]]

        article_scoring = np.array(article_scoring)
        logging.info("article num: {}".format(article_scoring.shape[0]))

    all_videos = np.load('xxx.npy',  allow_pickle=True)
    all_videos = torch.FloatTensor(all_videos)
    video_ids= np.load('xxx.npy')

    videos_index = {}

    video_scoring = []
    if len(video_ids)>0:
        #video_dataset = videoDataset(video_ids, all_videos,transform)
        video_dataset = videoDataset_np(all_videos,transform)
        video_dataloader = DataLoader(video_dataset,
                                     batch_size=512,
                                     shuffle = False,
                                     num_workers=2,
                                     #collate_fn = video_collate_fn,
                                     prefetch_factor = 1
                                     )
        logging.info('computing video...')

        from collections import defaultdict
        with torch.no_grad():
            for input_videos in tqdm(video_dataloader):
                # logging.info(input_ids.shape)
                input_items = defaultdict(list)
                input_items['txt_item'] = torch.LongTensor(input_items['txt_item']).cuda()
                pos_ids = [0] * input_videos.shape[0]
                input_items['vis_item'].extend(input_videos)
                input_items['vis_item'] = torch.stack(input_items['vis_item'], dim=0).cuda()
                input_items['pos_ids'].append(pos_ids)
                input_items['pos_ids'] = torch.LongTensor(input_items['pos_ids']).cuda()
                video_vec = model.item_encoder(input_items)
                video_vec = video_vec.to(torch.device("cpu")).detach().numpy()
                video_scoring.extend(video_vec) 

        video_scoring = np.array(video_scoring)

        logging.info("video scoring num: {}".format(video_scoring.shape[0]))

    if len(article_scoring)>0:
        index = len(article_index)
        whole_index = article_index.copy()
        print('article',index)
    else:
        whole_index = {}
        index = 0
    if len(video_ids)>0:
        for key, value in enumerate(video_ids):
            whole_index[value] = index+key
    print('whole',len(whole_index))
    
    def trans_to_nindex(nids, db_index):
        rr = []
        rr_index = {}
        index = 0
        for i in nids:
            rr.append(db_index[i])
            rr_index[i] = index
            index+=1
        return rr, rr_index      
    ids = np.load('xxx.npy', allow_pickle=True)
    rr, rr_index = trans_to_nindex(ids, whole_index)
    
    if len(video_ids)>0 and len(article_scoring)>0:
        whole_scoring = np.concatenate((article_scoring, video_scoring), axis=0)[rr]
    elif len(video_ids)>0:
        whole_scoring = video_scoring[rr]
    else:
        whole_scoring = article_scoring[rr]
    print('whole',whole_scoring.shape)    
    
    test_dataset = MFlowTorchDataset(whole_scoring, rr_index, mode='nip')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(),
                                                                   rank=hvd.rank())
    dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=512,
                            pin_memory=True, num_workers=4, collate_fn=smp_collate_fn)


    from metrics_whole_torch import  hr, auc_score, ndcg_score

    HR5 = []
    HR10 = []
    nDCG5 = []
    nDCG10 = []

    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
                                              '\t'.join(["{:0.8f}".format(i) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]
    
    whole_vecs = torch.FloatTensor(whole_scoring).cuda(non_blocking=True)
    for cnt, (log_vecs, log_mask, log_mask1, labels, mim_labels, pos_ids) in enumerate(dataloader):
        if args.enable_gpu:
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_mask = log_mask[:,1:].cuda(non_blocking=True)           
            log_mask1 = log_mask1.cuda(non_blocking=True)
            pos_ids = pos_ids.cuda(non_blocking=True)
            mim_labels = mim_labels.cuda(non_blocking=True)

        log_vec_cls = model.embeddings(log_vecs)
        user_vecs = model.user_encoder(log_vec_cls, log_mask)[0]
        indices = torch.where(log_mask1==1)
        user_vecs = user_vecs[indices]
        scores = torch.mm(user_vecs, whole_vecs.transpose(0,1))
        scores = scores+mim_labels
        poss = torch.argsort(torch.argsort(scores, dim=1), dim=1)+1
        
        for index, score, label in zip(
                range(len(labels)), poss, labels):  # batch,

            hr5 = hr(label, score, k=5)
            hr10 = hr(label, score, k=10)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)
            HR5.append(hr5)
            HR10.append(hr10)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

    print_metrics(hvd_rank, cnt * 512, get_mean([HR5, HR10, nDCG5, nDCG10]))
    
if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    print('testing...')
    samples = ['xxx']
    models = ['xxx']
    step = [772]
    for i, sample in enumerate(samples):
        print('testing'+sample)
        for ckpt in [i for i in range(1, 51)]:
            if i != 0:
                break
            print(str(ckpt))
            test(args, sample, models[i], str(ckpt), str(step[i]))
