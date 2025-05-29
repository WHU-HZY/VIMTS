# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.vision_transformer import PatchEmbed, Block
from .transformers.vision_transformer import Block
from .layers.GCN import gcn
from .util import IMTS_PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed_for_seq
import einops


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, history_len=24, patch_size=2, stride=2, pred_len=1, input_dim=32,
                 embed_dim=768, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mask_flag=False,
                 supports = None,gcn_config=None, lora_config=None, periodicity=None,mask_ratio=None,train_mode='fine-tune'):
        super().__init__()
        
        assert gcn_config != None and lora_config != None
        
        self.supports = supports
        
        self.periodicity = periodicity
        
        self.patch_size = patch_size
        
        self.mask_ratio = mask_ratio
        
        self.train_mode = train_mode
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.wo_gcn = gcn_config['wo_gcn']
        if self.wo_gcn:
            self.patch_embed = IMTS_PatchEmbed(input_dim=input_dim,embed_dim=embed_dim,bias=True) # w/o gcn
        else:
            self.patch_embed = IMTS_PatchEmbed(input_dim=input_dim+input_dim,embed_dim=embed_dim,bias=True) # gcn

        self.embed_dim = embed_dim

        self.decoder_dim = decoder_embed_dim

        # calculate num_patches from history_len and patch_size
        self.num_patches = int(ceil((history_len - patch_size) / stride) + 1)
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=True)  # fixed sin-cos embedding


        self.pred_patches = int(ceil((pred_len - patch_size) / stride) + 1)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, mask_flag=False, lora_config=lora_config)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        
        # --------------------------------------------------------------------------
        # gcc initialization
        self.supports_len = 0  
        if supports is not None:    
            self.supports_len += len(supports)
        
        nodevec_dim = gcn_config['node_dim']
        self.nodevec_dim = nodevec_dim
        
        N = gcn_config['N']
        self.N = N
        
        self.n_layer = gcn_config['n_layer']
        
        hid_dim = gcn_config['hid_dim']
        self.hid_dim = hid_dim
   
        if supports is None:
            self.supports = []
   
        # Node embeddings for dynamic support
        self.nodevec1 = nn.Parameter(torch.randn(N, nodevec_dim).cuda(), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(nodevec_dim, N).cuda(), requires_grad=True)
        
        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        for _ in range(self.n_layer):   
            self.nodevec_linear1.append(nn.Linear(hid_dim, nodevec_dim))
            self.nodevec_linear2.append(nn.Linear(hid_dim, nodevec_dim))
            self.nodevec_gate1.append(nn.Sequential(
				nn.Linear(hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))
            self.nodevec_gate2.append(nn.Sequential(
				nn.Linear(hid_dim+nodevec_dim, 1),
				nn.Tanh(),
				nn.ReLU()))

        self.supports_len +=1

        self.gnn_layers = nn.ModuleList() # gragh conv
        for _ in range(self.n_layer):
            self.gnn_layers.append(gcn(hid_dim, hid_dim, 0, support_len=self.supports_len, order=gcn_config['hop']))
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # # Channel Embedding
        # self.channel_embed = nn.Parameter(torch.zeros(self.N, decoder_embed_dim)) # (N, ttcn_dim)
        # # init 正态分布
        # self.channel_embed.data.copy_(torch.randn(self.N, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1 + self.pred_patches, 
                                                          decoder_embed_dim), requires_grad=True)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,pred_patches=self.pred_patches, mask_flag=mask_flag, lora_config=lora_config)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim + input_dim, input_dim, bias=True) # decoder to patch  decoded feature + time embedding --> prediction
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        # self.encode_grid_col = self.num_patches if periodicity is None else periodicity[1]
        # self.encode_grid_row = self.num_patches if periodicity is None else periodicity[0]
        
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (an`d freeze) pos_embed by sin-cos embedding
        pos_rows = int(self.periodicity//self.patch_size) if self.periodicity is not None else self.num_patches
        pos_cols = ceil(torch.tensor(self.num_patches/pos_rows)) if self.periodicity is not None else 1
        pos_embed = get_2d_sincos_pos_embed_for_seq(embed_dim=self.embed_dim, grid_size=(pos_rows,pos_cols), cls_token=True) # [BN, M, D]
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed[:self.num_patches+1, :]).float().unsqueeze(0))

        # initialize decoder_pos_embed by sin-cos embedding
        dec_pos_rows = int(self.periodicity//self.patch_size) if self.periodicity is not None else self.num_patches + self.pred_patches
        dec_pos_cols = ceil(torch.tensor((self.num_patches+self.pred_patches)/pos_rows)) if self.periodicity is not None else 1
        decoder_pos_embed = get_2d_sincos_pos_embed_for_seq(self.decoder_dim, (dec_pos_rows,dec_pos_cols), cls_token=True) # [BN, M+1, D]
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed)[:self.num_patches+1+self.pred_patches, :].float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        noise: [N, L]
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(round(L * (1 - mask_ratio)))
        
        if noise is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward(self, x):
        
        # GNN
        if not self.wo_gcn: # if use gcn
            x = self.GNN_Channel_Interaction(x)
        
        # Encoding
        x, mask, ids_restore = self.IMTS_encoder(x)
        
        # Decoding
        pred = self.IMTS_decoder(x, ids_restore)  # [N, L, p*p*3]
        
        return None, pred, mask

    def GNN_Channel_Interaction(self,x):
        # GNN
        BN, M, H = x.shape
        N = self.N
        B = BN // N
        x = x.reshape(-1,N,M,H)
        x_start = x.clone()
        for layer in range(self.n_layer):
            gnn = self.gnn_layers[layer]
            nodevec1 = self.nodevec1.view(1, 1, N, self.nodevec_dim).repeat(B, M, 1, 1)
            nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, N).repeat(B, M, 1, 1)
            x_gate1 = self.nodevec_gate1[layer](torch.cat([x, nodevec1.permute(0, 2, 1, 3)], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x, nodevec2.permute(0, 3, 1, 2)], dim=-1))
            x_p1 = x_gate1 * self.nodevec_linear1[layer](x) # (B, M, N, 10)
            x_p2 = x_gate2 * self.nodevec_linear2[layer](x) # (B, M, N, 10)
            nodevec1 = nodevec1 + x_p1.permute(0,2,1,3) # (B, M, N, 10)
            nodevec2 = nodevec2 + x_p2.permute(0,2,3,1) # (B, M, 10, N)

            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1) # (B, M, N, N) used
            new_supports = self.supports + [adp]
            x = gnn(x.permute(0,3,1,2), new_supports)
            x = x.permute(0, 2, 3, 1) # (B, N, M, F)
        x = x + x_start
        x = torch.cat((x_start,x), dim=-1)
        x = x.reshape(BN, M, -1)
        return x

    def IMTS_encoder(self,x):
        
        # embed patches
        x = self.patch_embed(x)
    
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking
        if self.train_mode == 'fine-tune':
            mask = ids_restore = None
        elif self.train_mode == 'pre-train' or self.train_mode == 'ttt':
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        else:
            raise ValueError("Unsupported train mode. Choose 'fine-tune' or 'pre-train' or 'ttt'.")

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks (with index)
        # for blk, gnn in zip(self.blocks, self.gnn_layers):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        return x, mask, ids_restore

    def IMTS_decoder(self, x, ids_restore):
        '''
        x: [BN, M, D], sequence
        mask: [BN, Pred], 1 is keep, 0 is remove,
        '''

        # embed tokens
        x = self.decoder_embed(x)

        # restore the masked tokens
        if self.train_mode == 'pre-train' or self.train_mode == 'ttt':
            mask_tokens_history = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens_history], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token [BN, M+1, D]

        else:
            # append mask tokens to sequence
            mask_tokens_predict = self.mask_token.repeat(x.shape[0], self.pred_patches, 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens_predict], dim=1)  # no cls token
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token [BN, M+1+pred_patches, D]

        # add pos embed
        x = x + self.decoder_pos_embed[:, :x.shape[1], :]
        
        # add channel embedding [N,D] --> [BN, M, D]
        # x = x + einops.rearrange(self.channel_embed, 'N D -> N 1 D').repeat(x.shape[0]//self.N, x.shape[1], 1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # # predictor projection
        # x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


def mae_vit_base_patch16_dec512d8b(**kwargs):
    norm = nn.LayerNorm
    # norm = ChannelLayerNorm
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(norm, eps=1e-6), **kwargs) # history_len=24, stride=2, patch_size=2
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) # history_len=24, stride=2, patch_size=2
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):  
    model = MaskedAutoencoderViT(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs) # history_len=24, stride=2, patch_size=2
    return model 


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
