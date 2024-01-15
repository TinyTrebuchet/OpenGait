import cv2 
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateBNNecks

max_frames_num = 256
max_contour_len = 512

class FeatureExtractor():

    def __init__(self, extra_feats=[], max_contour_len=512):
        self.feature_dim = 2
        assert self.feature_dim == len(extra_feats) + 2, f"Feature dimension mismatch. Expected {self.feature_dim} but got {len(extra_feats) + 2}"
        self.max_contour_len = max_contour_len

    def _extract_contour_from_img(self, img):
        # extract biggest outermost contour 
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        biggest_contour_idx = np.argmax([contour.size for contour in contours])
        contour = contours[biggest_contour_idx].squeeze()

        # reverse each point as (y,x) -> (x,y) and pad to max_contour_len
        contour = np.flip(contour.squeeze(), axis=1)
        num_padding =  self.max_contour_len - contour.shape[0]
        padding = np.zeros((num_padding, *contour.shape[1:]))
        contour_padded = np.vstack((contour, padding))

        # generate padding mask
        padding_mask = np.zeros(num_padding, dtype=bool)
        padding_mask[-num_padding:] = True

        return contour_padded, padding_mask

    def __call__(self, sils):
        b, s, h, w = sils.size()
        # extract features & generate padding mask
        feats = torch.empty((b, s, self.max_contour_len, self.feature_dim))
        pad_mask = torch.zeros((b, s, self.max_contour_len), dtype=torch.bool)
        for bi in range(b):
            for si in range(s):
                contour, pad_submask = self._extract_contour_from_img(sils[bi][si])
                feats[bi][si] = torch.from_numpy(contour)
                pad_mask[bi][si] = torch.from_numpy(pad_submask)
        return feats, pad_mask


class FactorizedEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward=2048):
        super(FactorizedEncoder, self).__init__()

        self.d_model = d_model

        self.spatial_pos_embedding = nn.Embedding(max_contour_len+1, d_model)
        self.temporal_pos_embedding = nn.Embedding(max_frames_num+1, d_model)

        self.spatial_cls_token = nn.Parameter(torch.randn(d_model))
        self.temporal_cls_token = nn.Parameter(torch.randn(d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, norm_first=True, batch_first=True)
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, contour_pad_mask):
        b, s, c, d = x.size()
        assert (d == self.d_model), f"Input dimension mismatch (d_model). Expected {self.d_model} but got {d}"

        spatial_cls_tokens = self.spatial_cls_token.expand((b, s, 1, d))
        x = torch.cat((spatial_cls_tokens, x), dim=2)
        x += self.spatial_pos_embedding[:(c+1)]

        x = rearrange(x, 'b s c d -> (b s) c d')
        x = self.spatial_encoder(x)[:,0]
        x = rearrange(x[:,0], '(b s) d -> b s d', b=b)

        temporal_cls_tokens = self.temporal_cls_token.expand((b, 1, d))
        x = torch.cat((temporal_cls_tokens, x), dim=1)
        x += self.temporal_pos_embedding[:(s+1)]

        x = self.temporal_encoder(x)[:,0]

        return x


class RGait(BaseModel):

    def __init__(self, *args, **kargs):
        super(RGait, self).__init__(*args, **kargs)
    
    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']

        self.FeatureExtractor = FeatureExtractor(extra_feats=[], max_contour_len=max_contour_len)

        self.Linear1 = nn.Linear(in_c[0], in_c[1])
        # ViT-Lite : https://arxiv.org/pdf/2104.05704.pdf
        self.FactorizedEncoder = FactorizedEncoder(d_model=in_c[1], 
                                                   num_layers=model_cfg['num_layers'], 
                                                   nhead=model_cfg['num_head'], 
                                                   dim_feedforward=model_cfg['dim_feedforward'])
        
        self.Head0 = nn.Linear(in_c[1], in_c[2])

        if 'SeparateBNNecks' in model_cfg.keys():
            self.BN_head = False
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        else:
            self.BN_head = True
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = nn.Linear(in_c[-1], class_num)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]
        del ipts

        with torch.no_grad():
            feats, pad_mask = self.FeatureExtractor(sils)
            feats = feats.to(device=torch.device("cuda", self.device))
            pad_mask = pad_mask.to(device=torch.device("cuda", self.device))
        del sils

        outs = self.Linear1(feats)
        outs = self.FactorizedEncoder(outs)
        gait = self.Head0(outs)

        if self.BN_head:    # Original Head
            bnft = self.Bn(gait)
            logi = self.Head1(bnft)
            embed = bnft
        else:               # BNNeck as Head
            bnft, logi = self.BNNecks(gait)
            embed = gait
        
        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval



        