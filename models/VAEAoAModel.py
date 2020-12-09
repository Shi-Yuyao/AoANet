#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Project     :  AoANet.py
@File        :  VAEAoAModel.py 
@Author      :  M.sc.SHI 
@Modify Time :  on 2020/12/7 at 19:28  
@Software    :  PyCharm Professional  
@Contact     :  <yukyewshek@outlook.com>
@License     :  (C)Copyright 1994-2020, VINCENT.S PRODUCTION Co.,Ltd.
@Version     :  0.1.0    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from models.AttModel import pack_wrapper, AttModel, Attention
from models.TransformerModel import LayerNorm, attention, clones, SublayerConnection, \
    PositionwiseFeedForward, PositionalEncoding, Embeddings


class MultiHeadedDotAttention(nn.Module):
    def __init__(self,
                 h,
                 d_model,
                 dropout=0.1,
                 scale=1,
                 project_k_v=1,
                 use_output_layer=1,
                 do_aoa=0,
                 norm_q=0,
                 dropout_aoa=0.3):
        super(MultiHeadedDotAttention, self).__init__()

        assert d_model * scale % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)  # 生成了三个in512 out512的fc

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            # 建立一个in1024 out1024的fc+GLU
            self.aoa_layer = nn.Sequential(nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU())
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:  # [query,value,key] size[batch_size,object_nums,dim]
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            # q,k,v分别经过fc，分成8个head
            query_, key_, value_ = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                                    for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch. x为ctx向量[5bz,8h,21obn,64dim]，self.attn为权重值[5,8,21,21]
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(self.dropout_aoa(torch.cat([x, query], -1)))
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


class VAE_AoA_Refiner_Layer(nn.Module):
    # size：feature维度；self_attn：多头注意力网络；feed_forward：位置前馈网络
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(VAE_AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)  # 残差网络模块
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 将KQV与mask通过多头注意力网络，再使用残差模块
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x  # 将上一步的输出与位置前馈网络使用残差模块连接


class VAE_AoA_Refiner_Core(nn.Module):
    def __init__(self, opt):
        super(VAE_AoA_Refiner_Core, self).__init__()
        # 建立多头注意力网络
        attn = MultiHeadedDotAttention(opt.num_heads,
                                       opt.rnn_size,
                                       project_k_v=1,
                                       scale=opt.multi_head_scale,
                                       do_aoa=opt.refine_aoa,
                                       norm_q=0,
                                       dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))

        # 建立Refiner_Layer,前馈网络就是fc,输出就是2048
        layer = VAE_AoA_Refiner_Layer(opt.rnn_size, attn,
                                      PositionwiseFeedForward(opt.rnn_size, 2048, 0.1)
                                      if opt.use_ff else None, 0.1)

        self.layers = clones(layer, 6)  # 将Refiner_Layer克隆6次
        self.norm = LayerNorm(layer.size)  # 建立layer normalization

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class VAE_AoA_Encoder_Core(nn.Module):
    def __init__(self, opt, size):
        super(VAE_AoA_Encoder_Core, self).__init__()
        # 初始化
        # 1.生成一个self Multi-head attention模块，以seq GT为输入
        # 2.生成一个带AoA的self Multi-head attention模块，以每一个ctx向量与image feature为输入
        # 3.建立一个MLP网络，以新的ctx向量与上一时刻生成的latent space z(t-1)为输入
        self.dropout = 0.1
        self.embed_layer = Embeddings(d_model=512, vocab=opt.vocab_size)
        self.position_encoding_layer = PositionalEncoding(d_model=512, dropout=0.1, max_len=opt.seq_length)
        self.attn1 = MultiHeadedDotAttention(h=8,
                                             d_model=512,
                                             project_k_v=0,
                                             use_output_layer=1,
                                             do_aoa=0,
                                             norm_q=1)
        self.attn2 = MultiHeadedDotAttention(h=8,
                                             d_model=512,
                                             do_aoa=1,
                                             norm_q=0)
        self.sublayer = clones(SublayerConnection(size, self.dropout), 1 + self.use_ff)

    def forward(self, refined_att_feats, seq_GT, latent_space_prior, mask=None):
        # 网络传输过程：
        # 1.将Seq GT通过self Multi-head attention，生成m个ctx向量，称为c(i)
        # 2.将每一个ctx向量分别与image feature做Multi-head attention，生成对应的分值，称为a(i)
        # 3.将a与ctx通过AoA模块，做一次attention on attention，生成新的ctx向量，称为new_c(i)
        # 4.将新的ctx向量与上一时刻生成的latent space z(t-1)通过一个MLP，生成均值与方差z(t)
        embedd_seq_GT = self.embed_layer(seq_GT)
        position = self.position_encoding_layer(seq_GT)
        x = embedd_seq_GT + position
        x = self.sublayer[0](x, lambda x: self.attn1(x, x, x, mask))
        x = self.sublayer[-1](x, self.feed_forward) if self.use_ff else x
        att = self.attn2(x,
                         refined_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model),
                         refined_att_feats.narrow(2, self.multi_head_scale * self.d_model,
                                                  self.multi_head_scale * self.d_model), mask)

        return att


class Latent_Space_Core(nn.Module):
    def __init__(self, opt):
        super(Latent_Space_Core, self).__init__()
        self.MLP_layer = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 2))

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, att_feats_seq, latent_sample_prior):
        x = torch.cat((att_feats_seq + latent_sample_prior), dim=1)
        mu, logvar = self.MLP_layer(x)
        return self.reparameterize(mu, logvar), mu, logvar


class VAE_AoA_Decoder_Core(nn.Module):
    def __init__(self, opt):
        super(VAE_AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'AoA')

        # 构建LSTMCell，input维度为512(word embedding)+512(Image features)+512(latent space)，hidden layer维度为512
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size + opt.latent_space_size,
                                    opt.rnn_size)  # we, fc, h^2_t-1
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        if self.decoder_type == 'AoA':
            # AoA layer
            self.att2ctx = nn.Sequential(
                nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'VAEAoA':
            # VAE AoA layer
            self.att2ctx = nn.Sequential(
                nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            # LSTM layer
            self.att2ctx = nn.LSTMCell(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            # Base linear layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size),
                                         nn.ReLU())

        if opt.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0,
                                                     scale=opt.multi_head_scale,
                                                     use_output_layer=0, do_aoa=0, norm_q=1)
        else:
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        # 1. 输入image features mean+latent sample current+word embedding current生成ht
        # 2. 将ht与image features attn共同输入multi-head attn得到hat a
        # 3. 将ht与hat a输入AoA模块生成ctx vector current
        # 4. 将ctx vector current通过linear+softmax作为output current，计算交叉熵Loss

        # state[0][1] is the context vector at the last step
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))

        if self.use_multi_head == 2:
            att = self.attention(h_att,
                                 p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model),
                                 # 取第二维的值，从multi_head_scale * d_model开始，到multi_head_scale * d_model结束
                                 p_att_feats.narrow(2,
                                                    self.multi_head_scale * self.d_model,
                                                    self.multi_head_scale * self.d_model),
                                 att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
        else:
            output = self.att2ctx(ctx_input)
            # save the context vector to state[0][1]
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        return output, state


class VAE_AoA_Black_Core(nn.Module):
    def __init__(self, opt):
        super(VAE_AoA_Black_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'AoA')

        # 构建LSTMCell，input维度为512(word embedding)+512(Image features)+512(latent space)，hidden layer维度为512
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size + opt.latent_space_size,
                                    opt.rnn_size)  # we, fc, h^2_t-1
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        if self.decoder_type == 'AoA':
            # AoA layer
            self.att2ctx = nn.Sequential(
                nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'VAEAoA':
            # VAE AoA layer
            self.att2ctx = nn.Sequential(
                nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size), nn.GLU())
        elif self.decoder_type == 'LSTM':
            # LSTM layer
            self.att2ctx = nn.LSTMCell(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size)
        else:
            # Base linear layer
            self.att2ctx = nn.Sequential(nn.Linear(self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size),
                                         nn.ReLU())

        if opt.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0,
                                                     scale=opt.multi_head_scale,
                                                     use_output_layer=0, do_aoa=0, norm_q=1)
        else:
            self.attention = Attention(opt)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        # 1. 输入image features mean+latent sample prior+word embedding prior生成ht
        # 2. 将ht通过MLP生成均值与方差z_black(t)，z_black(t)将作为下一时刻的black core的输入
        # 3. 输出生成均值与方差，将与Latent_Space_Core的均值与方差做KL Loss

        # state[0][1] is the context vector at the last step
        h_att, c_att = self.att_lstm(torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))

        if self.use_multi_head == 2:
            att = self.attention(h_att,
                                 p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model),
                                 # 取第二维的值，从multi_head_scale * d_model开始，到multi_head_scale * d_model结束
                                 p_att_feats.narrow(2,
                                                    self.multi_head_scale * self.d_model,
                                                    self.multi_head_scale * self.d_model),
                                 att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        if self.decoder_type == 'LSTM':
            output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
            state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
        else:
            output = self.att2ctx(ctx_input)
            # save the context vector to state[0][1]
            state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        return output, state


class VAEAoAModel(AttModel):
    def __init__(self, opt):
        super(VAEAoAModel, self).__init__(opt)
        self.num_layers = 2
        # mean pooling
        self.use_mean_feats = getattr(opt, 'mean_feats', 1)
        if opt.use_multi_head == 2:
            del self.ctx2att
            self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.multi_head_scale * opt.rnn_size)

        if self.use_mean_feats:
            del self.fc_embed
        if opt.refine:
            self.refiner = VAE_AoA_Refiner_Core(opt)
        else:
            self.refiner = lambda x, y: x
        self.core = VAE_AoA_Decoder_Core(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed att feats
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.refiner(att_feats, att_masks)

        if self.use_mean_feats:
            # meaning pooling
            if att_masks is None:
                mean_feats = torch.mean(att_feats, dim=1)
            else:
                mean_feats = (torch.sum(att_feats * att_masks.unsqueeze(-1), 1) / torch.sum(att_masks.unsqueeze(-1), 1))
        else:
            mean_feats = self.fc_embed(fc_feats)

        # Project the attention feats first to reduce memory and computation.
        p_att_feats = self.ctx2att(att_feats)

        return mean_feats, att_feats, p_att_feats, att_masks
