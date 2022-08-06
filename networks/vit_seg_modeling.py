# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
import pdb
# pdb.set_trace()


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight


# Pyramidal Convolution Channel Attention
class PCCA(nn.Module):
    def __init__(self, in_dim, reduction_dim, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):       # in_dim = 1024, reduction_dim = 512
        super(PCCA, self).__init__()

        self.features = nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
                nn.Conv2d(reduction_dim, reduction_dim // 4, kernel_size=pyconv_kernels[0], stride=stride,
                                 padding=pyconv_kernels[0]//2, dilation=1, groups=pyconv_groups[0], bias=False),
                nn.BatchNorm2d(reduction_dim // 4),
                nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(reduction_dim, reduction_dim // 4, kernel_size=pyconv_kernels[1], stride=stride,
                                 padding=pyconv_kernels[1]//2, dilation=1, groups=pyconv_groups[1], bias=False),
                nn.BatchNorm2d(reduction_dim // 4),
                nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(reduction_dim, reduction_dim // 4, kernel_size=pyconv_kernels[2], stride=stride,
                                 padding=pyconv_kernels[2]//2, dilation=1, groups=pyconv_groups[2], bias=False),
                nn.BatchNorm2d(reduction_dim // 4),
                nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim // 4, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim // 4),
                nn.ReLU(inplace=True)
        )
        self.se = SEWeightModule(reduction_dim // 4)

        self.weight_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_4 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_1.data.fill_(1.0)
        self.weight_2.data.fill_(1.0)
        self.weight_3.data.fill_(1.0)
        self.weight_4.data.fill_(1.0)
        self.softmax = nn.Softmax(dim=1)
        self.split_channel = reduction_dim // 4

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = nn.UpsamplingBilinear2d(scale_factor=2)(self.conv1(self.features(x)))
        x2 = nn.UpsamplingBilinear2d(scale_factor=2)(self.conv2(self.features(x)))
        x3 = nn.UpsamplingBilinear2d(scale_factor=2)(self.conv3(self.features(x)))
        x4 = self.conv4(x)

        x1_se = self.weight_1 * self.se(x1)
        x2_se = self.weight_2 * self.se(x2)
        x3_se = self.weight_3 * self.se(x3)
        x4_se = self.weight_4 * self.se(x4)
        # print(self.weight_1, self.weight_2, self.weight_3, self.weight_4)
        feats = torch.cat((x4, x1, x2, x3), dim=1)
        x_se = torch.cat((x4_se, x1_se, x2_se, x3_se), dim=1)

        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        return out


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)  # 224`0

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]      # (14, 14)
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])     # (1, 1)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)     # (16, 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])     # 14 * 14
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        # Embeddings
        self.patch_embeddings = Conv2d(in_channels=in_channels,     # 1024
                                       out_channels=config.hidden_size,     # 768
                                       kernel_size=patch_size,     # (1, 1)
                                       stride=patch_size)     # (1, 1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # PCCA branch
        self.pcca = PCCA(in_channels, 512)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        tail_x = x
        x1 = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))  (B, 768, 14, 14)
        x1 = x1.flatten(2)        # (B, 768, 196)
        x1 = x1.transpose(-1, -2)  # (B, n_patches, hidden)       (B, 196, 768)
        pcca_feat = self.pcca(x)

        embeddings = x1 + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features, pcca_feat, tail_x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvGRU_Cell(nn.Module):
    def __init__(self, channel_in, out_channel):    # 128, 64
        super().__init__()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_x_z = Conv2dReLU(channel_in,out_channel,kernel_size=3,stride=1, padding=1)
        self.conv_h_z = Conv2dReLU(out_channel,out_channel,kernel_size=3,stride=1, padding=1)
        self.conv_x_r = Conv2dReLU(channel_in,out_channel,kernel_size=3,stride=1, padding=1)
        self.conv_h_r = Conv2dReLU(out_channel,out_channel,kernel_size=3,stride=1, padding=1)
        self.conv = Conv2dReLU(channel_in,out_channel,kernel_size=3,stride=1, padding=1)
        self.conv_h = Conv2dReLU(out_channel,out_channel,kernel_size=3,stride=1, padding=1)
        self.conv_out = Conv2dReLU(out_channel,out_channel,kernel_size=3,stride=1, padding=1)

    def forward(self, x, h_0):
        z_t = torch.sigmoid(self.conv_x_z(x) + self.conv_h_z(h_0))
        r_t = torch.sigmoid((self.conv_x_r(x) + self.conv_h_r(h_0)))
        h_1_hat = torch.tanh(self.conv(x) + self.conv_h(torch.mul(r_t, h_0)))
        h_1 = torch.mul((1 - z_t), h_0) + torch.mul(z_t, h_1_hat)
        y = self.conv_out(h_1)
        return y


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features, pcca_feat, tail_x = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, 196, 768)
        return encoded, pcca_feat, attn_weights, features, tail_x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, with_gru=True, use_batchnorm=True):
        super().__init__()
        if skip_channels != 0:
            self.convgru = ConvGRU_Cell(in_channels, skip_channels)
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,        # [512, 256, 128, 64] + [512, 256, 64, 0]
            out_channels,           # (256, 128, 64, 16)
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,           # (256, 128, 64, 16)
            out_channels,           # (256, 128, 64, 16)
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.with_gru = with_gru

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if self.with_gru:
                skip = self.convgru(x, skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config, with_gru=True, with_pcca=True):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])  #  [512, 256, 128, 64]
        out_channels = decoder_channels     # (256, 128, 64, 16)
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0        # (512, 256, 64, 0)

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, with_gru) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.with_gru = with_gru
        self.with_pcca = with_pcca
        self.convgru_1 = ConvGRU_Cell(head_channels, head_channels)
        self.convgru_2 = ConvGRU_Cell(head_channels, head_channels)
        self.convgru = ConvGRU_Cell(head_channels, head_channels)
        self.cat_conv = Conv2dReLU(
                head_channels * 2,
                head_channels,
                kernel_size=1,
                padding=0,
                use_batchnorm=True,
        )

    def forward(self, hidden_states, pcca_feat, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)       # hidden_size --> 512
        if self.with_pcca:
            if self.with_gru:
                x = self.convgru(x, pcca_feat)
            else:
                x = self.cat_conv(torch.cat((x, pcca_feat), dim=1))

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class GPAMNet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, with_gru=True, with_pcca=True):
        super(GPAMNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)

        self.decoder = DecoderCup(config, with_gru, with_pcca)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, pcca_feat, attn_weights, features, tail_x = self.transformer(x)  # x: (B, n_patch, hidden)
        x = self.decoder(x, pcca_feat, features)       # include Conv_GRU
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


