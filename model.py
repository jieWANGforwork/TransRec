import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import torchvision.models
from layer import BertLayer
import time
from layer import BertOnlyMLMHead

def back_to_pos(txt_embd, vis_embed, position_ids, item_dim):
    output = []
    batch = position_ids.shape[0]
    logs = position_ids.shape[1]
    i = 0
    j = 0
    for u in range(batch):
        for l in range(logs):
            if position_ids[u, l] == 0:
                output.append(vis_embed[i])
                i += 1
            elif position_ids[u, l] == 1:
                output.append(txt_embd[j])
                j += 1
            elif position_ids[u, l] == -1:
                output.append(torch.Tensor([0]*item_dim).cuda())
    return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg:
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)#64, 200/1000
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):#batchsize, loglength, 1000
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)#batchsize, loglength, hidden_size
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)#batchsize, loglength, 1

        alpha = torch.exp(alpha)#batchsize, loglength, 1
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)#每个log乘以权重，batch,64,1
        x = torch.reshape(x, (bz, -1))  # (bz, 400) 64   user2,400
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        #       [bz, 20, seq_len, 20] x [bz, 20, 20, seq_len] -> [bz, 20, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)#先exp再乘mask
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        #       [bz, 20, seq_len, seq_len] x [bz, 20, seq_len, 20] -> [bz, 20, seq_len, 20]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, enable_gpu):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300 768
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        self.enable_gpu = enable_gpu

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None, seq_mask = None):
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        batch_size, seq_len, _ = Q.shape#(2 5, 64)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        if mask is not None:#(2, 5)
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len) #[bz, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)# attn_mask : [bz, 20, seq_len, seq_len]
        if seq_mask:
            seq_mask = 1 - torch.triu(torch.ones((batch_size, seq_len, seq_len), dtype=torch.uint8, device=mask.device), diagonal=1)
            seq_mask = seq_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            mask = mask.type_as(seq_mask) & seq_mask
            mask = mask.float()
        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)  # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  #self.layer_norm(output + residual)

class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 word_embedding_dim,
                 num_attention_heads,
                 query_vector_dim,
                 dropout_rate,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        # self.word_embedding = word_embedding
        self.bert_model = bert_model
        self.dropout_rate = dropout_rate
        self.multihead_attention = MultiHeadAttention(word_embedding_dim,
                                                      num_attention_heads, 20,
                                                      20, enable_gpu)
        self.additive_attention = AdditiveAttention(num_attention_heads * 20,
                                                    query_vector_dim)

    def forward(self, text, mask=None):

        batch_size, num_words = text.shape
        num_words = num_words // 3
        #text_ids = torch.narrow(text, 1, 0, num_words)
        #text_type = torch.narrow(text, 1, num_words, num_words)
        #text_attmask = torch.narrow(text, 1, num_words * 2, num_words)
        text_ids = text[: , 0:num_words]  # 1/3为text ID
        text_type = text[ : , num_words:2*num_words]  # 2/3,sequence type
        text_attmask = text[ : , 2*num_words:3*num_words]  # 3/3, mask
        word_emb = self.bert_model(input_ids=text_ids,attention_mask=text_attmask,token_type_ids=text_type)[0]

        text_vector = F.dropout(word_emb,
                                p=self.dropout_rate,
                                training=self.training)
        multihead_text_vector = self.multihead_attention(
            text_vector, text_vector, text_vector, mask)
        multihead_text_vector = F.dropout(multihead_text_vector,#400
                                          p=self.dropout_rate,
                                          training=self.training)
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(multihead_text_vector, mask)
        return text_vector

class TxtEncoder(torch.nn.Module):

    def __init__(self, args, bert_model):
        super(TxtEncoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 3,
            'abstract': args.num_words_abstract * 3,
            'body': args.num_words_body * 3,
            'category': 1,
            'domain': 1,
            'subcategory': 1
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum( list(self.attributes2length.values())[:list(self.attributes2length.keys()).index(key)])
            for key in self.attributes2length.keys()
        }
        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title']
        self.text_encoders = nn.ModuleDict({
            'title':
                TextEncoder(bert_model,
                            args.word_embedding_dim,
                            args.num_attention_heads, args.news_query_vector_dim,
                            args.drop_rate, args.enable_gpu)
        })

        self.newsname = [name for name in set(args.news_attributes) & set(text_encoders_candidates)]
        self.reduce_dim_linear = nn.Linear(args.num_attention_heads * 20,
                                           args.news_dim)
        if args.use_pretrain_news_encoder:
            self.reduce_dim_linear.load_state_dict(
                torch.load(os.path.join(args.pretrain_news_encoder_path,
                                        'reduce_dim_linear.pkl'))
            )

    def forward(self, news):

        text_vectors = [
            self.text_encoders['title'](
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]))
            for name in self.newsname
        ]

        all_vectors = text_vectors
        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:

            final_news_vector = torch.mean(
                torch.stack(all_vectors, dim=1),
                dim=1
            )

        # batch_size, news_dim
        final_news_vector = self.reduce_dim_linear(final_news_vector)
        return final_news_vector

class VisEncoder(torch.nn.Module):
    def __init__(self, args, resnet_model):
        super(VisEncoder, self).__init__()
        self.args = args
        self.model = nn.Sequential(*list(resnet_model.children()))[:-1]

        if args.item_dim!=512:
            self.reduce_dim_linear = nn.Linear(512, args.item_dim)
    def forward(self,  vis_feat):
        x = self.model(vis_feat)
        output = torch.flatten(x, 1)

        if self.args.item_dim!=512:
            output = self.reduce_dim_linear(output)
        #output = self.LayerNorm(output)
        #output = self.attnpool(x)
        return output


class ItemEncoder(torch.nn.Module):

    def __init__(self, args, bert_model, resenet_model):
        super(ItemEncoder, self).__init__()
        self.args = args
        self.txt_embeddings = TxtEncoder(args, bert_model)
        self.vis_embeddings = VisEncoder(args, resenet_model)
        #self.LayerNorm = nn.LayerNorm(args.item_dim)
        #self.item_pool = Projector(args)
       
    def forward(self, input_items):
        input_ids = input_items['txt_item']
        vis_feat = input_items['vis_item']
        position_ids = input_items['pos_ids']
        txt_embd = []
        vis_embed = []
        if min(vis_feat.shape) > 0:
            vis_embed = self.vis_embeddings(vis_feat)
        if min(input_ids.shape) > 0:
            txt_embd = self.txt_embeddings(input_ids)

        item_output = back_to_pos(txt_embd, vis_embed, position_ids, self.args.item_dim)
        item_output = torch.stack(item_output, dim=0)
        return item_output

class UserEncoderEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, config.user_log_length, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, log_vec, mim_mask=None):
        embeddings = log_vec
        batch_size, seq_len, _ = embeddings.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if mim_mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = mim_mask.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * w + mask_tokens * (1 - w)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        #embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.num_u_layers = args.user_layers
        if args.use_adapter:
            self.user_layer = nn.ModuleList(
            [BertLayer(args) if i<0 else BertLayer(args, args.use_adapter) for i in range(self.num_u_layers)]
        )
        else:
            self.user_layer = nn.ModuleList(
            [BertLayer(args) for _ in range(self.num_u_layers)]
        )
        if args.use_pool:
            self.pool = AdditiveAttention(
            args.item_dim, args.user_query_vector_dim)
    def forward(self, log_vec_cls, log_mask):

        hidden_states = log_vec_cls
        all_encoded_layers = []

        extended_log_mask = log_mask.unsqueeze(1).unsqueeze(2)
        extended_log_mask = extended_log_mask.to(dtype=next(self.parameters()).dtype)
        extended_log_mask = (1.0 - extended_log_mask) * -10000.0
        for layer_module in self.user_layer:
            hidden_states = layer_module(hidden_states, extended_log_mask)
            all_encoded_layers.append(hidden_states)
        if self.args.mim_task:
            return [hidden_states,  all_encoded_layers]
        elif self.args.use_pool:
            logging.info('user pool')
            user_vec = self.pool(hidden_states[:, 1:], log_mask[:,1:])
        else:
            #user_vec = hidden_states[ :,:1]
            user_vec = hidden_states[ :,:1].squeeze(dim=1)

        return user_vec

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.functional.relu
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.item_dim, config.vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        return hidden_states

class UserEncoderEmbeddings_NSP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, config.user_log_length, config.hidden_size))
        else:
            self.position_embeddings = None
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, log_vec, mim_mask=None):

        embeddings = log_vec
        batch_size, seq_len, _ = embeddings.size()

        if mim_mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            w = mim_mask.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * w + mask_tokens * (1 - w)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        if self.config.mode == 'train':
            embeddings = self.dropout(embeddings)

        return embeddings

class TransRec(torch.nn.Module):

    def __init__(self, args, bert_model, resnet_model):
        super(TransRec, self).__init__()
        self.args = args
        self.item_encoder = ItemEncoder(args, bert_model, resnet_model)
        self.embeddings = UserEncoderEmbeddings_NSP(args)
        if self.args.use_modal_embeddings:
            self.modal_embeddings = nn.Embedding(3, args.hidden_size)
        self.user_encoder = UserEncoder(args)
        if args.bce:            
            self.criterion = nn.BCEWithLogitsLoss()
        for m in [self.user_encoder.modules(),self.embeddings.modules()] :
                self.init_bert_weights(m)
                
        if self.args.from_scratch:
            logging.info('train from scratch')
            for m in self.modules():
                self.init_bert_weights(m)
            #self.apply(self.init_bert_weights)
        if self.args.use_adapter:
            logging.info('adapter')
            for m in self.user_encoder.modules():
                self.init_bert_weights(m) 
            logging.info('projector')
            #for m in self.projector.modules():
              #  self.init_bert_weights(m)
    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            logging.info("re-initialize linear.")
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            logging.info("re-initialize BertLayerNorm.")
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            logging.info("re-initialize bias.")

    def forward(self,
                input_items,
                log_items,
                log_mask,
                log_mask1,
                targets=None,
                compute_loss=True):

        item_vec = self.item_encoder(input_items)
        item_vec = item_vec.view(-1,  self.args.ppratio+self.args.npratio, self.args.item_dim)

        log_vec = self.item_encoder(log_items)
        log_vec = log_vec.view(-1, self.args.user_log_length, self.args.item_dim)
        log_vec_cls = self.embeddings(log_vec)
        if self.args.use_modal_embeddings:
            log_vec_cls = log_vec_cls + self.modal_embeddings((log_items['pos_ids']+1))    
        user_vector = self.user_encoder(log_vec_cls, log_mask)[0]
        indices = torch.where(log_mask1==1)
        user_vector = user_vector[indices]
        pos_score = torch.bmm(item_vec[:,:self.args.ppratio,], user_vector.unsqueeze(-1)).squeeze(dim=-1)
        neg_score = torch.bmm(item_vec[:,self.args.ppratio:,], user_vector.unsqueeze(-1)).squeeze(dim=-1)
        c = pos_score[:,0] - neg_score[:,0]

        if compute_loss:
            if self.args.bce:
                loss =  self.criterion(pos_score.view(-1),torch.FloatTensor([1]*(len(pos_score)*self.args.ppratio)).cuda(non_blocking=True)) 
                loss += self.criterion(neg_score.view(-1),torch.FloatTensor([0]*(len(pos_score)*self.args.npratio)).cuda(non_blocking=True)) 
            else:
                loss = -torch.log(torch.sigmoid(c)+1e-10).mean() 

            return loss, c
        else:
            return c
        


