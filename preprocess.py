import logging
from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re 
from utils import word_tokenize
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM

def read_article_bert(article_path, args, tokenizer, mode='train'):  # chineseBert的tokenizer
    article = {}
    article_index = {}
    index = 0
    logging.info('reading article...')

    if article_path=='xxx':
        article =np.load('xxx.npy',allow_pickle=True)
        article_index = np.load('xxx.npy',allow_pickle=True)
        article = article[()]
        article_index = article_index[()]
    with tf.io.gfile.GFile(article_path, "r") as f:
        i = 0
        for line in tqdm(f):
            splited = line.strip().split()
            doc_id = splited[0]
            title =splited[1]
            article_index[doc_id] = index
            index += 1

            if 'title' in args.article_attributes:
                title = tokenizer(title, max_length=args.num_words_title, \
                                  pad_to_max_length=True, truncation=True)
            else:
                title = []

            if 'abstract' in args.article_attributes:
                abstract = tokenizer(abstract, max_length=args.num_words_abstract, \
                                     pad_to_max_length=True, truncation=True)
            else:
                abstract = []

            if 'body' in args.article_attributes:
                body = tokenizer(body, max_length=args.num_words_body, \
                                 pad_to_max_length=True, truncation=True)
            else:
                body = []

            article[doc_id] = [title, abstract, body]
            i +=1
        print(i)
    logging.info(len(article_index))
    if mode == 'train' or 'test':
        return article, article_index
    else:
        assert False, 'Wrong mode!'


def get_doc_input_bert(article, article_index, args):
    #article_num = len(article) + 1
    article_num = len(article)

    if 'title' in args.article_attributes:
        article_title = np.zeros((article_num, args.num_words_title), dtype='int32')
        article_title_type = np.zeros((article_num, args.num_words_title), dtype='int32')
        article_title_attmask = np.zeros((article_num, args.num_words_title), dtype='int32')
    else:
        article_title = None
        article_title_type = None
        article_title_attmask = None

    if 'abstract' in args.article_attributes:
        article_abstract = np.zeros((article_num, args.num_words_abstract), dtype='int32')
        article_abstract_type = np.zeros((article_num, args.num_words_abstract), dtype='int32')
        article_abstract_attmask = np.zeros((article_num, args.num_words_abstract), dtype='int32')
    else:
        article_abstract = None
        article_abstract_type = None
        article_abstract_attmask = None

    if 'body' in args.article_attributes:
        article_body = np.zeros((article_num, args.num_words_body), dtype='int32')
        article_body_type = np.zeros((article_num, args.num_words_body), dtype='int32')
        article_body_attmask = np.zeros((article_num, args.num_words_body), dtype='int32')
    else:
        article_body = None
        article_body_type = None
        article_body_attmask = None

    for key in tqdm(article):
        title, abstract, body = article[key]
        doc_index = article_index[key]

        if 'title' in args.article_attributes:
            article_title[doc_index] = title['input_ids']
            article_title_type[doc_index] = title['token_type_ids']
            article_title_attmask[doc_index] = title['attention_mask']

        if 'abstract' in args.article_attributes:
            article_abstract[doc_index] = abstract['input_ids']
            article_abstract_type[doc_index] = abstract['token_type_ids']
            article_abstract_attmask[doc_index] = abstract['attention_mask']

        if 'body' in args.article_attributes:
            article_body[doc_index] = body['input_ids']
            article_body_type[doc_index] = body['token_type_ids']
            article_body_attmask[doc_index] = body['attention_mask']

    return article_title, article_title_type, article_title_attmask, \
           article_abstract, article_abstract_type, article_abstract_attmask, \
           article_body, article_body_type, article_body_attmask, \
