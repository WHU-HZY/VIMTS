import numpy as np
import torch

import os

from . import models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util
from collections import defaultdict
from timm.models.layers import Mlp

torch.autograd.set_detect_anomaly(True)
MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"

class VisionTS(nn.Module):

    def __init__(self, args, arch='mae_base', finetune_type='ln', ckpt_dir='../ckpt', load_ckpt=True):
        super(VisionTS, self).__init__()

        self.N = args.ndim
        self.M = args.npatch

        ### Intra-time series modeling ## 
		## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim-1)
        

        ## TTCN
        input_dim = 1 + args.te_dim
        ttcn_dim = args.hid_dim - 1
        self.ttcn_dim = ttcn_dim
        self.Filter_Generators = nn.Sequential(
				nn.Linear(input_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, input_dim*ttcn_dim, bias=True))
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))
        
        # Channel Embedding
        self.channel_embed = nn.Parameter(torch.zeros(self.N, ttcn_dim)) # (N, ttcn_dim)
        # init 正态分布
        self.channel_embed.data.copy_(torch.randn(self.N, ttcn_dim))
        
        # scalers for inverse norm
        # self.inverse_mean = nn.Parameter(torch.zeros(1,self.N), requires_grad=True)
        # self.inverse_std = nn.Parameter(torch.ones(1,self.N), requires_grad=True)

        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")
        
        self.history_len = args.history
        self.pred_len = args.pred_window
        self.stride = args.stride
        self.patch_size = args.patch_size
        self.hid_dim = args.hid_dim
        
        gcn_config = defaultdict(lambda: None)
        gcn_config['hid_dim'] = args.hid_dim
        gcn_config['hop'] = args.hop
        gcn_config['ndim'] = args.ndim
        gcn_config['node_dim'] = args.node_dim
        gcn_config['N'] = self.N
        gcn_config['n_layer'] = args.nlayer
        # gcc_parameter_dict['patch_size'] = self.patch_size
        # gcc_parameter_dict['hid_dim'] = self.hid_dim
        gcn_config['wo_gcn'] = args.wo_gcn
        
        
        self.train_mode = args.train_mode
        mae_config = {
            'history_len': self.history_len,
            'stride': self.stride,
            'patch_size': self.patch_size,
            'pred_len': self.pred_len,
            'mask_flag': args.mask_flag,
            'periodicity': args.periodicity,
            'mask_ratio': args.mask_ratio,
            'train_mode': self.train_mode,
            'input_dim': args.hid_dim,
        }
        
        lora_config = {
            'apply_lora': args.apply_lora,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'merge_weights': args.merge_weights
        }

        self.vision_model = MAE_ARCH[arch][0](gcn_config=gcn_config, lora_config=lora_config, **mae_config)
        
        self.pred_patches = self.vision_model.pred_patches
        self.history_patches = self.vision_model.num_patches

        # predictor time_embedding + reconstructed embedding ---> prediction
        # self.predictor = nn.Linear(args.te_dim + self.vision_model.decoder_dim, 1)
        self.predictor = nn.Sequential(
                nn.Linear(args.te_dim + self.vision_model.decoder_dim, args.te_dim + self.vision_model.decoder_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(args.te_dim + self.vision_model.decoder_dim, 1, bias=True))
        
        self.encoder_only = args.encoder_only
        if self.encoder_only:
            self.encode_proj = nn.Linear((self.vision_model.embed_dim) * self.history_patches, self.vision_model.embed_dim, bias=True)
            self.encoder_predictor = nn.Sequential(
                nn.Linear(args.te_dim + self.vision_model.embed_dim, args.te_dim + self.vision_model.embed_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(args.te_dim + self.vision_model.embed_dim, 1, bias=True))
        
        
        # self.predictor = Mlp(in_features=args.te_dim + self.vision_model.decoder_dim, out_features=1)
        ################################################################################################################

        if load_ckpt:
            ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
            if not os.path.isfile(ckpt_path):
                remote_url = MAE_DOWNLOAD_URL + MAE_ARCH[arch][1]
                util.download_file(remote_url, ckpt_path)
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                self.vision_model.load_state_dict(checkpoint['model'], strict=False)
            except:
                print(f"Bad checkpoint file. Please delete {ckpt_path} and redownload!")
        
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    
    # def update_config(self, context_len, pred_len, periodicity=1):
    #     self.num_patch = self.vision_model.num_patches
    #     self.context_len = context_len
    #     self.pred_len = pred_len
    #     self.periodicity = periodicity
        # self.grid_method = grid_method

        # self.pad_left = 0
        # self.pad_right = 0

        # if self.context_len % self.periodicity != 0:
        #     self.pad_left = self.periodicity - self.context_len % self.periodicity

        # if self.pred_len % self.periodicity != 0:
        #     self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        # input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        # self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        # if self.num_patch_input == 0:
        #     self.num_patch_input = 1
        # self.num_patch_output = self.num_patch - self.num_patch_input
        # adjust_input_ratio = self.num_patch_input / self.num_patch

        # interpolation = {
        #     "bilinear": Image.BILINEAR,
        #     "nearest": Image.NEAREST,
        #     "bicubic": Image.BICUBIC,
        # }[interpolation]

        # self.input_resize = util.safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        # self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))

        # self.output_resize = util.safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        # self.norm_const = norm_const
        
        # mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        # mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        # self.register_buffer("mask", mask.float().reshape((1, -1)))
        # self.mask_ratio = torch.mean(mask).item()
    
    def train_mode_update(self, train_mode): # 更新训练模式
        assert train_mode in ['fine-tune', 'pre-train', 'ttt'], "Invalid train mode"
        self.train_mode = train_mode
        self.vision_model.train_mode = train_mode
        print(f"Train mode updated to: {train_mode}")
        
    
    def LearnableTE(self, tt):
		# tt: (N*M*B, L, 1)
        tt = tt.reshape(tt.shape[0],tt.shape[1], 1)
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)
    
    def TTCN(self, X_int, mask_X):
		# X_int: shape (B*N*M, L, F)
		# mask_X: shape (B*N*M, L, 1)

        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int) # (N, Lx, F_in*ttcn_dim)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
        # normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (N, Lx, F_in*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1) # (N, Lx, ttcn_dim, F_in)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (N, ttcn_dim)
        h_t = torch.relu(ttcn_out + self.T_bias) # (N, ttcn_dim)
        return h_t
    
    
    # def forecasting(self, tp_to_predict=None, x=None, tt=None, tt_mask=None, tp_predict_mask=None, pred_patch_index=None, export_image=False, fp64=False):
    #         return self.forward(x, tt, tp_to_predict, tp_predict_mask, tt_mask, export_image, fp64, pred_patch_index)

    # def forward(self, x, tt=None, tp_to_predict=None, tp_predict_mask=None, tt_mask=None, \
    #     export_image=False, fp64=False, pred_patch_index=None):
    def forward(self, tp_to_predict=None, x=None, tt=None, tt_mask=None, tp_predict_mask=None, pred_patch_index=None, export_image=False, fp64=False):
        # Forecasting using visual model.
        # x: look-back window, size: [bs x context_len(t) x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len(t) x nvars]

        # forming the input features [B,M,L,N] --> [B*N*M,L,F]
        
        # inverse_mean = self.inverse_mean.repeat(x_mean.shape[0]*x_mean.shape[1]*x_mean.shape[2], 1)
        # inverse_std = self.inverse_std.repeat(x_mean.shape[0]*x_mean.shape[1]*x_mean.shape[2], 1)
        

        B, M, L_in, N =x.shape
        self.batch_size = B
        x = x.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
        
        # if self.training:
        #     x = x + torch.randn_like(x) * self.noise_std
        
        tt = tt.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
        tt_mask = tt_mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)

        # get time embedding and patch mask
        te_his = self.LearnableTE(tt) # (B*N*M, L, F_te)
        x = torch.cat([x, te_his], dim=-1)  # (B*N*M, L, F)
        mask_patch = (tt_mask.sum(dim=1) > 0).long() # (B*N*M, 1) # mask for the patch as features

        # TTCN for patch modeling
        x_patch = self.TTCN(x, tt_mask) # (B*N*M, hid_dim-1)
        
        # # add channel embedding
        x_patch = x_patch + einops.repeat(self.channel_embed, 'n d -> (b m n) d', b=B, m=M) # (B*N*M, hid_dim)
        
        x_patch = torch.cat([x_patch, mask_patch],dim=-1) # (B*N*M, hid_dim)
        x_patch = x_patch.view(self.batch_size, self.N, self.M, -1) # (B, N, M, hid_dim)
        B, N, M, D = x_patch.shape
        x = einops.rearrange(x_patch, 'b n m h -> (b n) m h') # [B, N, M, hid_dim] --> [B*N, M, hid_dim]
        # mask_patch = mask_patch.view(-1, self.M) # (B*N, M)

        # 4. Reconstruction [B*N, M, hid_dim] --> [B*N, M+P, hid_dim]
        # _, y, mask_ssl = self.vision_model(x)
        
        # 5. Forecasting & reconstruction
        if self.train_mode == 'fine-tune':
            
            if self.encoder_only:
                x = self.vision_model.GNN_Channel_Interaction(x)
                y, _, _= self.vision_model.IMTS_encoder(x)
                L_out = tp_to_predict.shape[1]
                te_pred = self.LearnableTE(tp_to_predict).unsqueeze(1).repeat(1, N, 1, 1) # (B, N, L_out, 10)
                latent_pred = einops.rearrange(y[:,1:,:], '(b n) l d -> b n (l d)', b=B)
                # print(latent_pred.shape)    
                latent_pred = self.encode_proj(latent_pred) # [B, N, 512]
                latent_pred = einops.rearrange(latent_pred, 'b n d -> b n 1 d')
                latent_pred = latent_pred.repeat(1, 1, L_out, 1)
                latent_decode = torch.cat([te_pred, latent_pred], dim=-1)
                latent_decode = einops.rearrange(latent_decode, 'b n l d -> b (n l) d')
                y = self.encoder_predictor(latent_decode) # [B, N * pred_len, 512+10] --> [B, N * L_out, 1]
                y = einops.rearrange(y.squeeze(), 'b (n l) -> b l n', b=B, n=N) # [B, N * L_out] --> [B, L_out, N]
                
            else:
                _, y, mask_ssl = self.vision_model(x)
                L_out = tp_to_predict.shape[1]
                te_pred = self.LearnableTE(tp_to_predict).unsqueeze(1).repeat(1, N, 1, 1) # (B, N, L_out, 10)
                latent_pred = einops.rearrange(y[:, -self.pred_patches:, :], '(b n) l d -> b n l d', b=B).unsqueeze(-2).repeat(1,1,1,L_out,1) # [B, N, P, L_out, 512]
                # latent_pred = y[:, -self.pred_patches:, :].unsqueeze(-2).repeat(1,1,L_out,1) # [B*N, pred_len, L_out, 512]  
                pred_patch_index = einops.rearrange(pred_patch_index, 'b l_out -> b 1 1 l_out 1').repeat(1, N, 1, 1, self.vision_model.decoder_dim) # [B, N, L_out, 512]
                latent_pred = latent_pred.gather(dim=2, index=pred_patch_index) # [B, N, L_out, 512]   
                
                te_pred = einops.rearrange(te_pred, 'b n l f -> b (n l) f') # [B, N, L_out, 10] --> [B, N*L_out, 10]
                latent_pred = einops.rearrange(latent_pred.squeeze(), 'b n l f -> b (n l) f') # [B, N, L_out, 512] --> [B, N*L_out, 512]
                # [N,ttcm_dim] -> [B, (N L) ,ttcm_dim]
                # channel_embedding = einops.repeat(self.channel_embed, 'n d -> b (n l) d', b=B, l=L_out) # (B*N*M, ttcn_dim)
                latent_decode = torch.cat([te_pred, latent_pred], dim=-1) # [B, N * L_out, 512+10]
                y = self.predictor(latent_decode) # [B, N * pred_len, 512+10] --> [B, N * L_out, 1]
                y = einops.rearrange(y.squeeze(), 'b (n l) -> b l n', b=B, n=N) # [B, N * L_out] --> [B, L_out, N]
            
        elif self.train_mode == 'pre-train' or self.train_mode == 'ttt':
            _, y, mask_ssl = self.vision_model(x)
            # number of the time stamps waiting for reconstruction
            L_out = M * L_in
            
            te_his = te_his.view(B*N, M, L_in, -1) # (B*N, M, L_in, F_te)
            
            # mask_ssl = einops.repeat(mask_ssl.unsqueeze(-1), 'b m 1 -> b m l', l=L_in) # (B*N, M, L_in)
            
            # te_his = self.LearnableTE(tp_to_predict).unsqueeze(1).repeat(1, N, 1, 1) # (B, N, L_out, 10)
            latent_rec = einops.repeat(y[:, :self.history_patches, :].unsqueeze(-2), 'b m 1 d -> b m l d', l=L_in) # (B*N, M, L_in, F_te)
            
            te_his = einops.rearrange(te_his, 'b n l d -> b (n l) d') # [B, N, L_out, 10] --> [B, N*L_out, 10]
            latent_rec = einops.rearrange(latent_rec, 'b n l d -> b (n l) d') # [B, N, L_out, F_te] --> [B, N*L_out, F_te]
            latent_decode = torch.cat([te_his, latent_rec], dim=-1) # [B, N * L_out, 512+10]
            y = self.predictor(latent_decode).squeeze() # [B, N * pred_len, 512+10] --> [B, N * L_out]

            return  y, mask_ssl.repeat(1,L_in)
                        
        else:
            raise ValueError("Unsupported train mode. Choose 'fine-tune' or 'pre-train' or 'ttt'.")
        
        return y


# resize image for reconstruction
# def forward(self, x, tt=None, tp_to_predict=None, tp_predict_mask=None, tt_mask=None, export_image=False, fp64=False):
#         # Forecasting using visual model.
#         # x: look-back window, size: [bs x context_len(t) x nvars]
#         # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
#         # return: forecasting window, size: [bs x pred_len(t) x nvars]

#         # forming the input features
#         B, M, L_in, N =x.shape
#         self.batch_size = B
#         X = X.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
#         tt = tt.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
#         tt_mask = tt_mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
#         te_his = self.LearnableTE(tt) # (B*N*M, L, F_te)
#         X = torch.cat([X, te_his], dim=-1)  # (B*N*M, L, F)
        

#         resolution_len = self.image_size

#         # 1. Normalization
#         means = x.mean(1, keepdim=True).detach() # [bs x 1 x nvars]
#         x_enc = x - means
#         stdev = torch.sqrt(
#             torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # [bs x 1 x nvars]
#         stdev /= self.norm_const
#         x_enc /= stdev
#         # Channel Independent [bs x nvars x seq_len]
#         # x_enc = einops.rearrange(x_enc, 'b s n -> b n s')
        
#         # 2. Segmentation
#         # x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate') # [b n s]
#         # x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)
#         # projection  [bs, nvars, context_len, resolution_len]
#         x_2d, new_tt = util.grid_time_interpolation(tt=tt, value=x_enc, resolution_len=resolution_len, dataset_len=48, method=self.grid_method, tt_mask=tt_mask)
#         new_value = F.pad(x_2d, (resolution_len*self.context_len-x_2d.shape[2], 0), mode='replicate') # [B D C*R]
#         new_value = einops.rearrange(new_value, 'b n (c r)-> (b n) 1 r c', r=resolution_len) # [B*D, 1, R, C]
    
#         # 3. Render & Alignment
#         x_resize = self.input_resize(x_2d).unsqueeze(1) # [bs x 1 x context_len x resolution_len]
#         masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
#         x_concat_with_masked = torch.cat([
#             x_resize, 
#             masked
#         ], dim=-1)
#         image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

#         # 4. Reconstruction
#         _, y, mask = self.vision_model(
#             image_input, 
#             mask_ratio=self.mask_ratio, noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
#         )
#         image_reconstructed = self.vision_model.unpatchify(y) # [(bs x nvars) x 3 x h x w]
        
#         # 5. Forecasting
#         y_grey = torch.mean(image_reconstructed, 1, keepdim=True) # color image to grey
#         y_segmentations = self.output_resize(y_grey) # resize back
#         y_flatten = einops.rearrange(
#             y_segmentations, 
#             '(b n) 1 f p -> b (p f) n', 
#             b=x_enc.shape[0], f=self.periodicity
#         ) # flatten
#         y = y_flatten[:, self.pad_left + self.context_len: self.pad_left + self.context_len + self.pred_len, :] # extract the forecasting window

#         # 6. Denormalization
#         y = y * (stdev.repeat(1, self.pred_len, 1))
#         y = y + (means.repeat(1, self.pred_len, 1))

#         # 7. get result [B, P, D] --> [B, T, D]
#         # start_tp tt每一行最后一个大于0的数
#         start_tp = torch.ceil(torch.max(tt, dim=-1)[0]*48)
#         # start_tp = torch.floor((tp_to_predict*48)[:,0])

#         y = util.get_prediction_from_reconstruction(y, tp_to_predict*48, start_tp, tp_predict_mask, tt_mask)

#         if export_image:
#             mask = mask.detach()
#             mask = mask.unsqueeze(-1).repeat(1, 1, self.vision_model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
#             mask = self.vision_model.unpatchify(mask)  # 1 is removing, 0 is keeping
#             # mask = torch.einsum('nchw->nhwc', mask)
#             image_reconstructed = image_input * (1 - mask) + image_reconstructed * mask
#             green_bg = -torch.ones_like(image_reconstructed) * 2
#             image_input = image_input * (1 - mask) + green_bg * mask
#             image_input = einops.rearrange(image_input, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
            
#             image_reconstructed = einops.rearrange(image_reconstructed, '(b n) c h w -> b n h w c', b=x_enc.shape[0])
#             return y, image_input, image_reconstructed
#         return y

