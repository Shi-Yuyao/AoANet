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
import torch.nn.functional as F


class MultiHeadedDotAttention(nn.Module):
    def __init__(self,
                 h, d_model, dropout=0.1, scale=1, project_k_v=1,
                 use_output_layer=1, do_aoa=0, norm_q=0, dropout_aoa=0.3):
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

        # Apply attention on all the projected vectors in batch.
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
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(VAE_AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


class VAE_AoA_Refiner_Core(nn.Module):
    """
    input:
        1. attended features: [batch_size, object_num, attention_dim]
           (original attention features dim pass feature embedding layer is attention_dim)
        2. attended masks: X
    output:
        refined attention features: [batch_size, object_num, attention_dim]
    """

    def __init__(self, opt):
        super(VAE_AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(opt.num_heads,
                                       opt.rnn_size,
                                       project_k_v=1,
                                       scale=opt.multi_head_scale,
                                       do_aoa=opt.refine_aoa,
                                       norm_q=0,
                                       dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))

        layer = VAE_AoA_Refiner_Layer(opt.rnn_size, attn,
                                      PositionwiseFeedForward(opt.rnn_size, 2048, 0.1)
                                      if opt.use_ff else None, 0.1)

        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class VAE_AoA_Encoder_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(VAE_AoA_Encoder_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


class VAE_AoA_Encoder_Core(nn.Module):
    """
    input:
        1. prepared refined attended features: [batch_size, object_num, 2 x attention_dim] for AoA
        2. sequence ground truth: [batch_size, max_sequence_length] for multi-head self-attention
        3. sequence masks: [batch_size, max_sequence_length] for multi-head self-attention's masks
    output:
        1. latent_space: [2, batch_size, max_sequence_length, latent_size]
            A for mu: [batch_size, latent_size] x max_sequence_length, merge as torch
            B for logvar: [batch_size, latent_size] x max_sequence_length, merge as torch
        2. latent_sample: [batch_size, latent_size] x max_sequence_length, as list
    """

    def __init__(self, opt, embed_layer):
        """
        1. 建立multi-head self-attention模块(VAE_AoA_Encoder_Layer x 6), 对sequence GT做自编码, 生成ctx vector[c1,c2,...,cn]
            input:
                sequence GT: [batch_size, max_sequence_length, word_embed_dim]
                PS. word_embed_dim会决定rnn_size, 因为会送入lstm中
            output:
                ctx vector ci: [batch_size, word_embed_dim]
                (ctx vector c: ci x max_sequence_length, as list)
        2. 建立一个带AoA的multi-head attention模块, 为ctx vector嵌入图像信息, 生成new ci
            input:
                ctx vector ci: [batch_size, word_embed_dim]
                prepared refined image attended feature A': [batch_size, object_num, 2 x attention_dim]
            output:
                new ctx vector new_ci: [batch_size, word_embed_dim]
                (new ctx vector c: new_ci x max_sequence_length, as list)
        3. 建立一个MLP模块, 为new ctx vector生成latent space, 时序生成
            (MLP模块包含有: fc层; mu计算层; logvar计算层)
            input:
                ctx vector ci: [batch_size, word_embed_dim]
                prior latent sample z_i-1: [batch_size, latent_size]
                PS. latent_size会决定rnn_size, 因为会送入lstm中
            output:
                mu_encoder mu_i: [batch_size, latent_size]
                logvar_encoder logvar_i: [batch_size, latent_size]
                (mu: mu_i x max_sequence_length, logvar: logvar_i x max_sequence_length, mu & logvar
                will merge as torch latent_space: [2, batch_size, max_sequence_length, latent_size])
        4. 对mu与logvar重采样, 为decoder生成该时刻的latent sample, 时序生成
            input:
                mu_encoder mu_i: [batch_size, latent_size]
                logvar_encoder logvar_i: [batch_size, latent_size]
            output:
                latent sample z_i: [batch_size, latent_size]
                (latent sample z: z_i x max_sequence_length, as list)

        Parameters
        ----------
        embed_layer: 因为与decoder解码时共用一个word embedding layer，所以特别传入
        """

        super(VAE_AoA_Encoder_Core, self).__init__()
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.latent_size = opt.latent_space_size

        # embedding layer+position encoding
        self.embed = embed_layer
        self.position_encoded = PositionalEncoding(d_model=opt.rnn_size, dropout=0.1, max_len=opt.seq_length)

        # multi-head self-attention for seq GT
        attn = MultiHeadedDotAttention(opt.num_heads,
                                       opt.rnn_size,
                                       project_k_v=1,
                                       scale=opt.multi_head_scale,
                                       do_aoa=1,
                                       norm_q=0,
                                       dropout_aoa=getattr(opt, 'dropout_aoa', 0.3))
        layer = VAE_AoA_Encoder_Layer(opt.rnn_size,
                                      attn,
                                      PositionwiseFeedForward(opt.rnn_size, 2048, 0.1)
                                      if opt.use_ff else None, 0.1)
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

        # multi-head attention for seq GT and refined feats
        self.att_p_refined_feats = MultiHeadedDotAttention(opt.num_heads,
                                                           opt.rnn_size,
                                                           project_k_v=0,
                                                           scale=opt.multi_head_scale,
                                                           use_output_layer=0,
                                                           do_aoa=1, norm_q=1)

        # MLP for latent sample
        self.MLP_layer = nn.Sequential(nn.Linear(opt.rnn_size + opt.latent_space_size, opt.mlp_hidden_size),
                                       nn.LayerNorm(opt.mlp_hidden_size),
                                       nn.LeakyReLU())
        self.MLP_layer_mu = nn.Linear(opt.mlp_hidden_size, opt.latent_space_size)
        self.MLP_layer_logvar = nn.Linear(opt.mlp_hidden_size, opt.latent_space_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, refined_p_att_feats, seq_GT, masks_padding, masks_seq=None):

        embedd_seq_GT = self.embed(seq_GT)
        x = self.position_encoded(embedd_seq_GT)

        for layer in self.layers:
            x = layer(x, masks_padding)
        c = self.norm(x)

        new_c = []
        for i in range(c.size(1)):
            c_i = c[:, i, :]
            new_c_i = self.att_p_refined_feats(c_i,
                                               refined_p_att_feats.narrow(
                                                   2, 0, self.multi_head_scale * self.d_model),
                                               refined_p_att_feats.narrow(
                                                   2, self.multi_head_scale * self.d_model,
                                                      self.multi_head_scale * self.d_model),
                                               mask=masks_seq)
            new_c.append(new_c_i)

        # init latent sample for the first time step, (batch_size,latent dim)
        latent_sample_prior = torch.randn(seq_GT.size(0), self.latent_size).cuda()

        # A save mu, B save log_var; (2, batch_size, seq_len, latent dim)
        latent_space = torch.zeros(2, seq_GT.size(0), seq_GT.size(1), self.latent_size).cuda()

        latent_sample = list()
        for index, j in enumerate(new_c):
            mlp_input = torch.cat((j, latent_sample_prior), dim=1)
            mlp_output = self.MLP_layer(mlp_input)
            mu = self.MLP_layer_mu(mlp_output)
            log_var = self.MLP_layer_logvar(mlp_output)
            latent_space[0, :, index] = mu
            latent_space[1, :, index] = log_var
            latent_sample_prior = self.reparameterize(mu, log_var)
            latent_sample.append(latent_sample_prior)

        # add a random latent sample for token <END>
        latent_sample.append(torch.randn(seq_GT.size(0), self.latent_size).cuda())

        return latent_space, latent_sample


class VAE_AoA_Decoder_Core(nn.Module):
    """
    input:
        0. hidden state: (h: [2, batch_size, rnn_size], c: [2, batch_size, rnn_size])
        1. prepared refined attended features: [batch_size, object_num, 2 x attention_dim] for AoA
        2. refined attended features mean: [batch_size, attention_dim] for LSTM
        3. context word last time step: [batch_size, rnn_size] for LSTM, plus to mean
               (saved in hidden state h dim2)
        4. -TRAIN- latent sample from encoder current time step: [batch_size, latent_size] for LSTM
           -TEST- latent sample from blackbox current time step: [batch_size, latent_size] for LSTM
        5. -TRAIN- embed ground truth word last time step: [batch_size, word_embed_dim] for LSTM
           -TEST- embed output word last time step: [batch_size, word_embed_dim] for LSTM
    output:
        1. context word current time step: [batch_size, rnn_size]
        2. hidden state current time step: (h: [2, batch_size, rnn_size], c: [2, batch_size, rnn_size])
    """

    def __init__(self, opt):
        """
        1. 建立一个LSTM模块，对输入的信息做编码，生成context vector c_t
            input:
                latent sample current time step z_t: [batch_size, latent_size]
                embed word last time step gt w_t-1 or output c_t-1: [batch_size, word_embed_dim]
                refined attended features mean A: [batch_size, attention_dim]
                context word last time step c_t-1: [batch_size, rnn_size], plus to mean
                (saved in hidden state h dim2)
            output:
                hidden state h_t: [batch_size, rnn_size]
        2. 建立一个multi-head attention模块，h_t与prepared refined image features A'做attention，生成 attend features
            input:
                prepared refined attended features A’: [batch_size, object_num, 2 x attention_dim]
                hidden state h_t: [batch_size, rnn_size]
            output:
                attend features a_t: [batch_size, rnn_size]
        3. 建立一个AoA网络，attend features与hidden state h_t做AoA, 生成该时刻的context vector c_t
            input:
                attend features a_t: [batch_size, rnn_size]
                hidden state h_t: [batch_size, rnn_size]
            output:
                context vector c_t: [batch_size, rnn_size]
                (context vector c_t pass linear+softmax as final output for calculating CE Loss)
        """

        super(VAE_AoA_Decoder_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_multi_head = opt.use_multi_head
        self.multi_head_scale = opt.multi_head_scale
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.decoder_type = getattr(opt, 'decoder_type', 'VAEAoA')

        # input dim: word_embed_dim(w_t-1) + attention_dim(mean) + latent_size(z_t), hidden layer dim: rnn_size
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size + opt.latent_space_size, opt.rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        if opt.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0,
                                                     scale=opt.multi_head_scale,
                                                     use_output_layer=0, do_aoa=0, norm_q=1)
        else:
            self.attention = Attention(opt)

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

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def forward(self, xt, zt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
        # state[0][1] is the context vector at the last step, xt is shifted right gt etc.label
        h_att, c_att = self.att_lstm(torch.cat([xt, zt, mean_feats + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))

        # use value from dim2，from multi_head_scale * d_model to multi_head_scale * d_model as end
        if self.use_multi_head == 2:
            att = self.attention(h_att,
                                 p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model),
                                 p_att_feats.narrow(2, self.multi_head_scale * self.d_model,
                                                    self.multi_head_scale * self.d_model),
                                 att_masks)
        else:
            att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        ctx_input = torch.cat([att, h_att], 1)
        output = self.att2ctx(ctx_input)

        # save the context vector to state[0][1]
        state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

        if self.out_res:
            # add residual connection
            output = output + h_att

        output = self.out_drop(output)
        return output, state


class VAE_AoA_Black_Core(nn.Module):
    """
    input:
        0. hidden state: (h: [2, batch_size, rnn_size], c: [2, batch_size, rnn_size])
        1. refined attended features mean: [batch_size, attention_dim] for LSTM
        2. embed output word last time step: [batch_size, rnn_size] for LSTM
        3. -TRAIN-
           -TEST- latent sample from blackbox current time step: [batch_size, latent_size] for LSTM
        5. -TRAIN- embed ground truth word last time step: [batch_size, word_embed_dim] for LSTM
           -TEST- embed output word last time step: [batch_size, word_embed_dim] for LSTM
    output:
        latent sample
            -TRAIN- for KL divergence Loss between encoder
            -TEST- provide current time step latent sample for decoder
    """

    def __init__(self, opt):
        """
        1. 建立一个LSTM模块，对输入的信息做编码，在test阶段为decoder生成当前时刻的latent sample
            input:
                embed word last time step c_t-1: [batch_size, word_embed_dim]
                refined attended features mean A: [batch_size, attention_dim]
                latent sample last time step z_t-1: [batch_size, latent_size]
            output:
                hidden state h_t: [batch_size, rnn_size]
        2. 建立一个MLP模块，对LSTM生成的h_t进行编码，生成一个latent space
            (MLP模块包含有: fc层; mu计算层; logvar计算层)
            input:
                hidden state h_t: [batch_size, rnn_size]
            output:
                mu_blackbox mu_t: [batch_size, latent_size]
                logvar_blackbox logvar_t: [batch_size, latent_size]
        3. 对mu与logvar重采样, 为decoder生成该时刻的latent sample
            input:
                mu_blackbox mu_t: [batch_size, latent_size]
                logvar_blackbox logvar_t: [batch_size, latent_size]
            output:
                latent sample z_i: [batch_size, latent_size]
        """
        super(VAE_AoA_Black_Core, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.d_model = opt.rnn_size
        self.use_ctx_drop = getattr(opt, 'ctx_drop', 0)
        self.out_res = getattr(opt, 'out_res', 0)
        self.latent_size = opt.latent_space_size

        # input dim: word_embed_dim(w_t-1) + attention_dim(mean) + latent_size(z_t), hidden layer dim: rnn_size
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size + opt.latent_space_size, opt.rnn_size)
        self.out_drop = nn.Dropout(self.drop_prob_lm)

        # MLP layer for latent sample
        self.MLP_layer = nn.Sequential(nn.Linear(opt.rnn_size, opt.mlp_hidden_size),
                                       nn.LayerNorm(opt.mlp_hidden_size),
                                       nn.LeakyReLU())
        self.MLP_layer_mu = nn.Linear(opt.mlp_hidden_size, opt.latent_space_size)
        self.MLP_layer_logvar = nn.Linear(opt.mlp_hidden_size, opt.latent_space_size)

        if self.use_ctx_drop:
            self.ctx_drop = nn.Dropout(self.drop_prob_lm)
        else:
            self.ctx_drop = lambda x: x

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_prior, mean_feats, z_prior, state):
        # TODO, if blackbox LSTM use c_t-1 add to mean feature as same as Decoder
        # state[0][1] is the context vector at the last step
        # h, c = self.att_lstm(torch.cat([x_prior, z_prior, mean_feats], 1), (state[0][0], state[1][0]))
        # state = (torch.stack((h, state[1][1])), torch.stack((c, state[1][1])))
        h, c = self.att_lstm(torch.cat([x_prior, z_prior, mean_feats], 1), state)
        state = (h, c)

        mlp_output = self.MLP_layer(h)
        mu = self.MLP_layer_mu(mlp_output)
        log_var = self.MLP_layer_logvar(mlp_output)
        z = self.reparameterize(mu, log_var)

        return z, mu, log_var, state


class VAEAoAModel(AttModel):
    def __init__(self, opt):
        super(VAEAoAModel, self).__init__(opt)
        self.latent_space_size = opt.latent_space_size
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
        # vocab_size+1为9488，即增加<END>
        # 重写AttModel的self.embed，采用与transformer一样的结果，且使得encoder与decoder的embed layer一致
        self.embed = nn.Sequential(Embeddings(d_model=opt.rnn_size, vocab=opt.vocab_size + 1),
                                   nn.ReLU(),
                                   nn.Dropout(0.5))
        self.encoder = VAE_AoA_Encoder_Core(opt, self.embed)
        self.blackbox = VAE_AoA_Black_Core(opt)
        self.core = VAE_AoA_Decoder_Core(opt)

    def init_hidden_blackbox(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(bsz, self.rnn_size),
                weight.new_zeros(bsz, self.rnn_size))

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        """
        Parameters
        ----------
        fc_feats: X
        att_feats: [batch_size, object_nums, attention_dims]
        att_masks: X

        Returns
        -------
        mean_feats: [batch_size, attention_dims]
        att_feats: [batch_size, object_nums, attention_dims]
        p_att_feats: [batch_size, object_nums, 2 x attention_dims]
        att_masks: X
        """
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

    def _forward(self, fc_feats, att_feats, seq, masks, att_masks=None):
        batch_size = fc_feats.size(0)
        state_decoder = self.init_hidden(batch_size)
        state_blackbox = self.init_hidden_blackbox(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)
        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        # 将masks转换为与seqGT匹配的shape,由于原始mask会优先保留1(在算loss时,需要增加1位<END>，会正数,而这里与loss里的转换不同，是倒数)
        masks = masks[:, -seq[:, 1:-1].shape[1]:]
        latent_space_encoder, latent_sample_encoder = self.encoder(pp_att_feats, seq[:, 1:-1], masks)
        z_blackbox_init = torch.randn(batch_size, self.latent_space_size).cuda()
        latent_space_blackbox = torch.zeros(2, batch_size, seq.size(1) - 2, self.latent_space_size).cuda()
        decoder_x_prior = list()
        latent_sample_blackbox = list()
        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            zt = latent_sample_encoder[i]
            # 'it' contains a word index
            output, state_decoder, x_prior = self._get_logprobs_state(it, zt, p_fc_feats,
                                                                      p_att_feats, pp_att_feats,
                                                                      p_att_masks, state_decoder)
            outputs[:, i] = output

            if i == 0:
                xt = self.embed(it)
                zt_blackbox, mu_blackbox, log_var_blackbox, state_blackbox = self.blackbox(xt,
                                                                                           p_fc_feats,
                                                                                           z_blackbox_init,
                                                                                           state_blackbox)
                latent_space_blackbox[0, :, i] = mu_blackbox
                latent_space_blackbox[1, :, i] = log_var_blackbox
                latent_sample_blackbox.append(zt_blackbox)
            elif i == seq.size(1) - 2:
                continue
            else:
                zt_blackbox, mu_blackbox, log_var_blackbox, state_blackbox = self.blackbox(decoder_x_prior[-1],
                                                                                           p_fc_feats,
                                                                                           latent_sample_blackbox[-1],
                                                                                           state_blackbox)
                latent_space_blackbox[0, :, i] = mu_blackbox
                latent_space_blackbox[1, :, i] = log_var_blackbox
                latent_sample_blackbox.append(zt_blackbox)
            decoder_x_prior.append(x_prior)
        return [outputs, latent_space_encoder, latent_space_blackbox]

    def _get_logprobs_state(self, it, zt, fc_feats, att_feats, p_att_feats, att_masks, state):
        xt = self.embed(it)
        output, state = self.core(xt, zt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state, output

    def _sample_get_logprobs_state(self, it, zt, state_blackbox, fc_feats, att_feats, p_att_feats, att_masks,
                                   state_decoder):
        xt = self.embed(it)
        zt, mu, log_var, state_blackbox = self.blackbox(xt, fc_feats, zt, state_blackbox)

        output, state_decoder = self.core(xt, zt, fc_feats, att_feats, p_att_feats, state_decoder, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state_decoder, state_blackbox, zt

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()  # [batch,max_len]建立一个max_length长度的空tensor，存放生成的word
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)  # 同理，存放生成word的Logprobs
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state_decoder = self.init_hidden(beam_size)
            state_blackbox = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k + 1].expand(
                *((beam_size,) + p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:  # input <bos>

                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)
                    zt = torch.randn(beam_size, self.latent_space_size).cuda()
                # 生成第一个word的logprobs, state，即初始化
                logprobs, state_decoder, state_blackbox, zt = self._sample_get_logprobs_state(it, zt,
                                                                                              state_blackbox,
                                                                                              tmp_fc_feats,
                                                                                              tmp_att_feats,
                                                                                              tmp_p_att_feats,
                                                                                              tmp_att_masks,
                                                                                              state_decoder)

            self.done_beams[k] = self.beam_search(state_decoder, state_blackbox, zt,
                                                  logprobs, tmp_fc_feats, tmp_att_feats,
                                                  tmp_p_att_feats,
                                                  tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
