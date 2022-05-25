import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
import torch
print(torch.cuda.device_count())
print(torch.cuda.is_available())
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
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from datetime import datetime
from dataloader import TransRecTorchDataset, smp_collate_fn, _all_videos_lmdb
import time
import torchvision
from paras import bert_finetuneset, transform, resnet_finetuneset

def save_model(model_path, ep, cnt, model, optimizer, loss, accuary):
    ckpt_path = os.path.join(model_path, f'epoch-{ep + 1}-{cnt + 1}.pt')
    torch.save(
        {
            'epoch': ep + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.data / (cnt + 1),
            'accuracy': accuary / (cnt + 1),
        }, ckpt_path)
    logging.info(f"Model saved to {ckpt_path}")

def train(args, model_path):
    if args.enable_hvd:
        import horovod.torch as hvd
        hvd.init()
        logging.info(
            f"hvd_size:{hvd.size()}, hvd_rank:{hvd.rank()}, hvd_local_rank:{hvd.local_rank()}"
        )
        hvd_size = hvd.size()
        hvd_rank = hvd.rank()
        hvd_local_rank = hvd.local_rank()
        if args.enable_gpu and torch.cuda.is_available():
            torch.cuda.set_device(hvd_local_rank)
        print(hvd_size, hvd_rank, hvd_local_rank)
 
    tokenizer = AutoTokenizer.from_pretrained('/bert/bert-base-chinese')
    config = AutoConfig.from_pretrained('/bert/bert-base-chinese', output_hidden_states=True)
    bert_model = BertModel(config=config)
    resnet_model = resnet18(pretrained=False)
    resnet_model.load_state_dict(torch.load('/resnet/resnet18-5c106cde.pth'))

    for name, param in bert_model.named_parameters():
        if name not in bert_finetuneset:
            logging.info(name)
            param.requires_grad = False
    for name, param in resnet_model.named_parameters():
        if name not in resnet_finetuneset:
            logging.info(name)
            param.requires_grad = False
    if hvd.rank()>0:
        time.sleep(5.000)    
    #all_videos = _all_videos_lmdb()
    video_ids = np.load('/articleclient_mix_seq_filter50k_vitems.npy')

    videos_index = {}
    def trans_to_videoindex(nids):
        rr_index = {}
        index = 0
        for i in nids:
            rr_index[i] = index
            index+=1
        return rr_index 
    if len(video_ids)>0:
        videos_index = trans_to_videoindex(video_ids)

    video = True
    if video:
        all_videos = np.load('/articleclient_mix_seq_filter50k_video.npy',  allow_pickle=True)
    else:
        all_videos = []
    all_videos = torch.FloatTensor(all_videos)
    
    logging.info('all_videos loaded')
    if hvd.rank()>0:
        time.sleep(5.000)
    article, article_index = read_article_bert('/articleclient/articleclient_article_50k', args, tokenizer)

    article_title, article_title_type, article_title_attmask, \
    article_abstract, article_abstract_type, article_abstract_attmask, \
    article_body, article_body_type, article_body_attmask= get_doc_input_bert(
        article, article_index, args)  # index对应序列
    if hvd.rank()>0:
        time.sleep(5.000)
    txt_item_doc = np.concatenate([article_title, article_title_type, article_title_attmask], axis=1)  # 按列拼接，列变多
    txt_item_doc = torch.LongTensor(txt_item_doc)
    logging.info('article loaded')

    model = TransRec(args, bert_model, resnet_model)
    if args.enable_gpu:
        model = model.cuda()
        if hvd.rank()>0:
            time.sleep(5.000)
    if hvd.rank()==0 and args.use_pretrained:
        check_path = 'XXX.pt'
        checkpoint = torch.load(check_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        pre_trained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        trained_dict = list(set(model_dict.keys()).difference(set(pre_trained_dict.keys())))
        logging.info('required grad')
        logging.info(trained_dict)
        logging.info('Model loaded from {}'.format(check_path))

    pre_params = []
    other_params = []
    for name, para in model.named_parameters():
        if para.requires_grad:
            if 'item_encoder' in name:
                logging.info(name)
                pre_params += [para]
            else:
                logging.info(name)
                other_params += [para]
    params = [
        {'params': pre_params, 'lr': 1e-4},
        {'params': other_params, 'lr': 1e-4}
        ]
    optimizer = optim.Adam(params)

    if args.resume and hvd.rank()==0:
        path_checkpoint = 'xxx.pt'
        logging.info(path_checkpoint)
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('resume model loaded...')

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Average)
        if hvd.rank()>0:
            time.sleep(5.000)

    lines = np.load('/articleclient/articleclient_mix_seq_filter50k.npy',allow_pickle=True)
    _clicks = []
    _o_line = []
    for line in lines:
        _clicks.append(line[:-2])
        _o_line.append(line)
    Loss = []
    train_dataset = MFlowTorchDataset(txt_item_doc, article_index, all_videos, videos_index, _clicks, _o_line , transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(),
                                                                    rank=hvd.rank())
    for ep in range(0, 50):
        logging.info('start loading data...')

        train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler,
                                  num_workers=6, pin_memory=True, collate_fn=smp_collate_fn,
                                      prefetch_factor = 2)
        logging.info('Training...')
        loss = 0.0
        accuary = 0.0
        model = model.train()
        for cnt, (log_items, log_mask, input_items, log_mask1) in enumerate(train_dataloader):#最后注意log_mask
            if cnt > args.max_steps_per_epoch:
                break

            if args.enable_gpu:
                log_items['txt_item'] = log_items['txt_item'].cuda(non_blocking=True)
                log_items['vis_item'] = log_items['vis_item'].cuda(non_blocking=True)
                log_items['pos_ids'] = log_items['pos_ids'].cuda(non_blocking=True)
                log_mask = log_mask.cuda(non_blocking=True)
                log_mask1 = log_mask1.cuda(non_blocking=True)
                input_items['txt_item'] = input_items['txt_item'].cuda(non_blocking=True)
                input_items['vis_item'] = input_items['vis_item'].cuda(non_blocking=True)
                input_items['pos_ids'] = input_items['pos_ids'].cuda(non_blocking=True)

            bz_loss, y_hat = model(input_items, log_items, log_mask, log_mask1)
            loss += bz_loss.data.float()
            bz_acc = (torch.sum(y_hat>0)/len(y_hat)).data.float()
            accuary += bz_acc
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            if hvd_rank == 0 and cnt % 10 == 0:
                logging.info(
                    '[{}] Ed: {}, batch:{}, batch_loss:{}, batch_acc:{},  epoch:{} train_loss: {:.5f}, acc: {:.5f}'.format(
                        hvd_rank, (cnt + 1) * args.batch_size, cnt + 1, bz_loss, bz_acc, ep + 1, loss.data / (cnt + 1),
                                  accuary / (cnt + 1)))
            if hvd_rank == 0 and (cnt+1) % args.save_steps == 0:
                save_model(model_path, ep, cnt+1, model, optimizer, loss, accuary)
            time.sleep(0.003)
        logging.info('epoch, loss, accuracy')
        print(ep + 1, loss / (cnt + 1), accuary / (cnt + 1))

        if hvd_rank == 0:
            Loss.append(loss.data / (cnt + 1))
            save_model(model_path, ep, cnt+1, model, optimizer, loss, accuary)

    Loss0 = np.array(Loss)
    np.save('loss.npy', Loss0)
    logging.info('loss saved')

if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    if args.resume:
        model_path = 'XXX'
        print('start resume training...')
    else:
        model_path = 'xxx' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path(model_path).mkdir(parents=True, exist_ok=True)
    train(args, model_path)
