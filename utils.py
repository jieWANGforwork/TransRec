import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import sys
import torch
import numpy as np
import argparse
import re

def init_hvd_cuda(enable_hvd=True, enable_gpu=True):
    hvd = None
    if enable_hvd:
        import horovod.torch as hvd

        hvd.init()
        logging.info(
            f"hvd_size:{hvd.size()}, hvd_rank:{hvd.rank()}, hvd_local_rank:{hvd.local_rank()}"
        )

    hvd_size = hvd.size() if enable_hvd else 1
    hvd_rank = hvd.rank() if enable_hvd else 0
    hvd_local_rank = hvd.local_rank() if enable_hvd else 0

    if enable_gpu:
        torch.cuda.set_device(hvd_local_rank)

    return hvd_size, hvd_rank, hvd_local_rank


def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")


def acc(y_true, y_hat):
    tot = y_true.shape[0]
    hit = torch.sum(y_hat > 0)
    return hit.data.float() * 1.0 / tot
def acc_msk(y_hat):
    tot = len(y_hat)
    hit = torch.sum(y_hat > 0)
    return hit.data.float() * 1.0 / tot

def acc_nsp(pos_scores,neg_scores):
    tot = pos_scores.shape[0]*pos_scores.shape[1]
    hit = torch.sum(pos_scores > neg_scores)
    #print(hit)
    return hit.data.float() * 1.0 / tot
    #return hit

def acc_mim(y_true, y_hat):# bz*seq+1, bz*seq+1*dim
    #y_hat = y_true.view(-1, y_hat.shape[2])
    #y_true = y_true.view(y_true.shape[0]*y_true.shape[1])
    y_pre = torch.argmax(y_hat, dim=2)#bz*seq+1
    s = torch.eq(y_pre, y_true)
    #logging.info(y_pre)
    #logging.info(y_true)
    tot = torch.sum(y_true!=-100)
    #logging.info(tot)
    hit = torch.sum(s)
    #logging.info(hit)
    return hit.data.float() * 1.0 / tot

def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    print(os.listdir(directory))
    if len(os.listdir(directory))==0:
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    print(ckpt_path)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None
