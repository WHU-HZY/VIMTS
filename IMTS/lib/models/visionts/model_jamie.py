import numpy as np
import torch

import os

from . import gat, models_mae
import einops
import torch.nn.functional as F
from torch import nn
from PIL import Image
from . import util, gat

MAE_ARCH = {
    "mae_base": [models_mae.mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [models_mae.mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [models_mae.mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}

MAE_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/mae/visualize/"



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim, bias=True)
        )
        self.ln1 = nn.LayerNorm(embed_dim) 
        self.ln2 = nn.LayerNorm(embed_dim) 

    def forward(self, x, key=None, value=None):
        if key == None:
            key = x
        if value == None:
            value = x
        attn_out = self.attn(query=x, key=key, value=value)[0]

        x = self.ln1(x + attn_out) 
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)  
        return x


class VisionTS(nn.Module):

    def __init__(self, args, arch='mae_base', finetune_type='ln', ckpt_dir='../ckpt', load_ckpt=True):
        super(VisionTS, self).__init__()

        self.N = args.ndim
        self.M = args.npatch

        ################################################################################################################
        # ## GAT

        # feature construction：
            # value + time + channel
        self.ce = nn.Parameter(torch.randn(self.N, args.hid_dim))
        self.de = nn.Linear(1, args.hid_dim)
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, args.te_dim-1)

        self.gat = gat.GAT(args.hid_dim * 2 + args.te_dim, args.hid_dim)

        ################################################################################################################
        # ## MAE
        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch: {arch}. Should be in {list(MAE_ARCH.keys())}")
        
        self.history_len = args.history
        self.pred_len = args.pred_window
        self.stride = args.stride
        self.patch_size = args.patch_size
        self.hid_dim = args.hid_dim

        self.vision_model = MAE_ARCH[arch][0](\
            history_len=self.history_len, \
                stride=self.stride, \
                    patch_size=self.patch_size,\
                        pred_len=self.pred_len,\
                            mask_flag=args.mask_flag,)
        
        self.pred_patches = self.vision_model.pred_patches
        self.history_patches = self.vision_model.num_patches

        ################################################################################################################
        # ## Transformer predictor
        self.num_blocks = 8
        self.att_dim = 128
        
        self.pre_channel_tb = nn.ModuleList([TransformerBlock(args.hid_dim, 8) for _ in range(self.num_blocks)])
        
        self.qe = nn.Linear(args.te_dim, self.att_dim)
        self.ke = nn.Linear(self.vision_model.decoder_dim, self.att_dim)
        self.ve = nn.Linear(self.vision_model.decoder_dim, self.att_dim)
        
        # Define MultiHeadAttention layer
        self.pos_time_tb = nn.ModuleList([TransformerBlock(self.att_dim, 8) for _ in range(self.num_blocks)])
        self.pos_channel_tb = nn.ModuleList([TransformerBlock(self.att_dim, 8) for _ in range(self.num_blocks)])

        # Define Linear Layer for prediction
        self.predictor = nn.Linear(self.att_dim, 1, bias=True)
        
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
    
    def forecasting(self, tp_to_predict=None, x=None, tt=None, tt_mask=None, tp_predict_mask=None, pred_patch_index=None, export_image=False, fp64=False):
            return self.forward(x, tt, tp_to_predict, tp_predict_mask, tt_mask, export_image, fp64, pred_patch_index)

    def forward(self, x, tt=None, tp_to_predict=None, tp_predict_mask=None, tt_mask=None, \
        export_image=False, fp64=False, pred_patch_index=None):
        '''
        # Forecasting using visual model.
        # x: look-back window, size: [bs x context_len(t) x nvars]
        # fp64=True can avoid math overflow in some benchmark, like Bitcoin.
        # return: forecasting window, size: [bs x pred_len(t) x nvars]
        '''
        ################################################################################################################
        # 1. forming the input features
        B, M, L_in, N =x.shape
        self.batch_size = B
        x = x.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
        tt = tt.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
        tt_mask = tt_mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)

        ################################################################################################################
        # 2. get feature embedding and patch mask
        mask_patch = (tt_mask.sum(dim=1) > 0).long() # (B*N*M, 1) # mask for the patch as features
        mask_patch = mask_patch.view(-1, M) # (B*N, M)

        # time embedding
        te_his = self.LearnableTE(tt) # (B*N*M, L, te_dim)
        de_his = self.de(x) # (B*N*M, L, hid_dim)
        ce_his = self.ce.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, M, L_in, 1, 1)  # 扩展到 (B, M, L, N, hidden_dim)
        ce_his = ce_his.permute(0, 3, 1, 2, 4).reshape(-1, L_in, self.hid_dim) # (B*N*M, L, hid_dim)
        x = torch.cat([te_his, ce_his, de_his], dim=-1) # (B*N*M, L_in, F)

        x = x.view(B, N, M, L_in, -1).permute(0, 2, 1, 3, 4).reshape(B*M, N*L_in, -1) # (M, B, N * L_in, -1)


        ################################################################################################################
        # 3. GAT node info exchange
        # # 计算边的数量 F
        # F = (L * (L - 1) // 2) * N

        # 生成所有 L1 < L2 的组合, 生成 k 的张量
        C = torch.combinations(torch.arange(L_in), r=2)
        k = torch.arange(N)

        # 生成 source 和 target 下标
        source = C[:, 0].unsqueeze(1) * N + k.unsqueeze(0)
        target = C[:, 1].unsqueeze(1) * N + k.unsqueeze(0)
        source = source.flatten()
        target = target.flatten()

        # 将 source 和 target 下标组合成 edge_index
        edge_index = torch.stack([source, target], dim=0)  # shape [2, F]

        # Add additional edges from global node to all nodes
        global_node_idx = L_in * N
        additional_source = torch.arange(L_in * N, dtype=torch.long)
        additional_target = torch.full((L_in * N,), fill_value=global_node_idx, dtype=torch.long)
        additional_edges = torch.stack([additional_source, additional_target], dim=0)  # shape [2, L*N]

        # Concatenate original and additional edges
        edge_index = torch.cat([edge_index, additional_edges], dim=1)  # shape [2, F + L*N]

        # Calculate global node features using mean aggregation
        
        global_node = torch.mean(x, dim=1, keepdim=True)  # Shape: [B*M, 1, N]
        # Concatenate global node features
        x = torch.cat((x, global_node), dim=1)  # Shape: [B*M, L*N + 1, N]

        x = self.gat(x, edge_index)

        x = x[:, -1, :].reshape(B, M, -1)  # [B, M, hid_dim]

        ################################################################################################################
        # 4. Reconstruction 
        _, y, _ = self.vision_model(x, mask_patch)  # [B, M, hid_dim] -> [B, M+P, hid_dim]
        
        ################################################################################################################
        # 5. Forecasting
        L_out = tp_to_predict.shape[1]
        te_pred = self.LearnableTE(tp_to_predict).unsqueeze(1).repeat(1, N, 1, 1) # (B, N, L_out, 10)

        # Slice y to get the last self.pred_patches elements
        y_sliced = y[:, -self.pred_patches:, :]  # Shape: [b, p, d]

        # Introduce n and l_out dimensions and repeat
        latent_pred = einops.repeat(y_sliced, 'b p d -> b n p l_out d', n=N, l_out=L_out)  # Shape: [b, n, p, l_out, d]

        # Adjust pred_patch_index dimensions
        pred_patch_index = einops.repeat(pred_patch_index, 'b l_out -> b n 1 l_out 1', n=N)  # Shape: [b, n, 1, l_out, 1]

        # Expand pred_patch_index to match the feature dimension
        pred_patch_index = pred_patch_index.expand(-1, -1, -1, -1, y_sliced.shape[-1])  # Shape: [b, n, 1, l_out, d]

        # Gather the latent predictions
        latent_pred = latent_pred.gather(dim=2, index=pred_patch_index).squeeze(2)  # Shape: [b, n, l_out, d]
        
        # Tranformer Predictor
        query = self.qe(te_pred.view(B * N, L_out, -1))  # (B * N, L_out, hid_dim)
        key = self.ke(latent_pred.view(B * N, L_out, -1))  # (B * N, L_out, hid_dim)
        value = self.ve(latent_pred.view(B * N, L_out, -1)) # (B * N, L_out, hid_dim))

        # Apply MultiHeadAttention
        z = query
        for i in range(len(self.pos_time_tb)):
            z = self.pos_time_tb[i](z, key, value)
            z = z.view(B, N, L_out, -1).permute(0, 2, 1, 3).reshape(B * L_out, N, -1) # (B * L_out, N, hid_dim)
            
            z = self.pos_channel_tb[i](z)
            z = z.view(B, L_out, N, -1).permute(0, 2, 1, 3).reshape(B * N, L_out, -1) # (B* N, L_out, hid_dim)
            
        z = z.view(B, N, L_out, -1).permute(0, 2, 1, 3).reshape(B * L_out, N, -1) # (B* L_out, N, hid_dim)

        # Apply MLP for prediction
        y = self.predictor(z)  # [B * L_out, N, 1]
        y = y.squeeze().view(B, L_out, N)  # [B, L_out, N]

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

