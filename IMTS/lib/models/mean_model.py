import torch.nn as nn
import torch
import torch.nn.functional as F

class MeanModel(nn.Module):
    def __init__(self):
        super(MeanModel, self).__init__()

    def forecasting(self, tp_to_predict=None, x=None, tt=None, tt_mask=None, tp_predict_mask=None, export_image=False, fp64=False):
        return self.forward(x, tt, tp_to_predict, tp_predict_mask, tt_mask, export_image, fp64)

    def forward(self, x, tt=None, tp_to_predict=None, tp_predict_mask=None, tt_mask=None, export_image=False, fp64=False):
        Output_T = tp_to_predict.size(1)
        return x.sum(dim=1, keepdim=True).repeat(1, 1, Output_T, 1) /(1e-8+tt_mask.sum(dim=1, keepdim=True).repeat(1, 1, Output_T, 1))	
