import argparse
import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
    parser.add_argument('--txt_model_name', type=str, default='bert-base-chinese')
    parser.add_argument("--from_scratch", type=utils.str2bool, default=False)
    parser.add_argument("--resume", type=utils.str2bool, default=False)
    parser.add_argument("--resume_path", type=str, default="/ceph/11329/jojiewang/qb/user4k/model2021-12-23_22-12-44")
    parser.add_argument("--test_model_path", type=str, default="/ceph/11329/jojiewang/qb/user4k/model2021-12-23_22-12-44")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    
    parser.add_argument("--root_data_dir", type=str,
                        default= "/group/30003/jojiewang/newsclient/data")
    parser.add_argument("--train_data_dir", type=str,
                        default="/ceph/11329/jojiewang/qqbrowser/pre300k420k")
    parser.add_argument("--test_data_dir", type=str, default='flickr_small')
    parser.add_argument('--mind_dataset', type=str, default='/home/wangjie/data/test/mind')
    parser.add_argument("--flickr_dataset",type=str, default='flickr_small')
    parser.add_argument('--txt_train_data_dir', type=str,default='/group/30003/jojiewang/newsclient/data')
    parser.add_argument('--img_train_data_dir', type=str,default='/group/30003/jojiewang/newsclient/data/video')

    parser.add_argument('--txt_item_test_data_dir',type=str,default='/ceph/11329/jojiewang/qqbrowser/pre4k23k')
    parser.add_argument("--train_dir",type=str,default='mind_train_sample_*.csv')
    parser.add_argument("--test_dir", type=str, default='valid_sample.txt')
    parser.add_argument("--filename_pat", type=str, default='train_sample_mflow.csv',
                        choices=['mind_train_sample_*.csv', 'behaviors_*.tsv', 'train_samples_*.csv', 'test_samples_*.csv', 'train_sample_mflow.csv'])


    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--npratio", type=int, default=5)
    parser.add_argument("--ppratio", type=int, default=5)

    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--enable_hvd", type=utils.str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--user_layers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)  # his: 0.00001,0.01,0.001:135 0.01;135后0.001 1e-3
    parser.add_argument("--pre_lr", type=float, default=2e-6)  # his: 0.00001 2e-6
    parser.add_argument("--news_attributes",type=str,nargs='+',default=['title'],
                        choices=['title', 'abstract', 'body', 'category', 'domain', 'subcategory'])
    parser.add_argument("--num_words_title", type=int, default=32)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--num_words_uet", type=int, default=16)
    parser.add_argument("--num_words_bing", type=int, default=26)
    parser.add_argument("--user_log_length",type=int,default=25)

    parser.add_argument("--word_embedding_dim", type=int, default=768)
    parser.add_argument("--use_padded_news_embedding", type=utils.str2bool, default=False)
    parser.add_argument("--padded_news_different_word_index", type=utils.str2bool,default=False)
    parser.add_argument("--news_dim", type=int, default=256)
    parser.add_argument('--item_dim', type=int, default=256)
    parser.add_argument("--news_query_vector_dim", type=int, default=200)
    parser.add_argument("--user_query_vector_dim", type=int, default=200)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--user_num_attention_heads", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--intermediate_size", type=int, default=128)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_act", type=str, default='gelu', choices=['relu', 'gelu', 'swish'])
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)

    parser.add_argument("--user_log_mask", type=utils.str2bool, default=True)
    parser.add_argument("--seq_mask", type=utils.str2bool, default=False)
    parser.add_argument("--use_pool", type=utils.str2bool, default=False)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--do_lower_case", type=utils.str2bool, default=True)
    parser.add_argument('--img_size', type=int, default=224)
    
    parser.add_argument("--mim_task", type=utils.str2bool, default=True)
    parser.add_argument("--use_mask_token", type=utils.str2bool, default=False)
    parser.add_argument("--use_absolute_position_embeddings", type=utils.str2bool, default=True)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--use_modal_embeddings", type=utils.str2bool, default=False)
    parser.add_argument("--pretrain_model_dir", type=str, default='/group/30003/jojiewang/user4k/adaptermodel')
    parser.add_argument("--pretrain_resume", type=utils.str2bool, default=False)
    parser.add_argument("--pretrain_resume_path", type=str, default="/pretrainmodel2021-12-23_22-12-44")
    parser.add_argument("--is_decoder", type=utils.str2bool, default=False)
    parser.add_argument("--vocab_size", type=int, default=48931)
    parser.add_argument("--bce", type=utils.str2bool, default=True)

    args = parser.parse_args()

    # logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
