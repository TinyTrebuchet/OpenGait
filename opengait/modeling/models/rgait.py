import cv2 
import math
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateBNNecks

max_frames_num = 128
max_contour_len = 128

def np2t(arr, device):
    t = torch.from_numpy(arr)
    return t.to(device=torch.device('cuda', device))

class FeatureExtractor():

    # resize arr to give size along given axis by trimming or padding
    # returns resized array & padding mask
    def _resize(self, arr, sz, axis=0):
        batch, og_sz, feat = arr.shape[:axis], arr.shape[axis], arr.shape[axis+1:]
        pad_mask = np.zeros((*batch, sz), dtype=bool)
        if arr.shape[axis] >= sz:
            resized_arr = np.take(arr, range(sz), axis=axis)
            return resized_arr, pad_mask
        else:
            padding_len = sz - og_sz
            padding = np.zeros((*batch, padding_len, *feat))
            resized_arr = np.append(arr, padding, axis=axis)
            assert resized_arr.shape == (*batch, sz, *feat), "Sanity check!"
            pad_mask[..., -padding_len:] = True
            return resized_arr, pad_mask
        
    def _extract_sil_contour(self, step=1, pad=False, method=cv2.CHAIN_APPROX_NONE):
        # extract biggest outermost contour 
        img = img.astype('uint8')
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, method)
        if len(contours) > 0:
            biggest_contour_idx = np.argmax([contour.size for contour in contours])
            contour = contours[biggest_contour_idx].squeeze()
            if contour.ndim < 2: # squeeze() drops the first axis if only 1 contour point exists
                contour = np.expand_dims(contour, axis=0)
            contour = np.flip(contour[::step], axis=-1) # flip coords from (y,x) to (x,y)
        else:
            contour = np.empty((0,2))
        assert contour.ndim == 2 and contour.shape[-1] == 2, f"Invalid contour shape recieved {contour.shape}"
        if pad:
            contour = self._resize(contour, max_contour_len, axis=0)
        return contour

class GLExtractor(FeatureExtractor):

    def __init__(self, max_contour_len=512):
        self.max_contour_len = max_contour_len
    
    def extract_sils_contour(self, sils, step=1):
        b, s, h, w = sils.shape
        contours = np.empty((b,s,self.max_contour_len,2), dtype=sils.dtype)
        pad_masks = np.empty((b,s,self.max_contour_len), dtype=bool)
        for bi in range(b):
            for si in range(s):
                contours[bi][si], pad_masks[bi][si] = self._extract_sil_contour(sils[bi][si], step, pad=True)
        return (contours, pad_masks)

    def extract_local_feats(self, sils, contours=None):
        centroids = np.expand_dims(np.mean(contours, axis=-2), axis=-2)
        dists = np.sqrt(np.square(contours - centroids).sum(axis=-1, keepdims=True))
        return np.concatenate((contours, dists), axis=-1)

    def extract_global_feats(self, sils, contours=None):
        bin_sils = (sils > 0).astype(sils.dtype)   # b s h w

        centroids = np.mean(contours, axis=-2)
        areas = np.expand_dims(bin_sils.sum(axis=(-1,-2)), axis=-1)
        
        wcs, hcs = np.cumsum(bin_sils, axis=-1), np.cumsum(bin_sils, axis=-2)
        left = wcs.argmin(axis=-1).min(axis=-1, keepdims=True)
        right = wcs.argmax(axis=-1).max(axis=-1, keepdims=True)
        top = hcs.argmin(axis=-2).min(axis=-1, keepdims=True)
        bottom = hcs.argmax(axis=-2).max(axis=-1, keepdims=True)
        height, width = (bottom - top).astype(sils.dtype), (right - left).astype(sils.dtype)

        return np.concatenate((centroids, areas, height, width), axis=-1)
    
    def __call__(self, sils, local_step=1):
        contours, pad_masks = self.extract_sils_contour(sils, step=2)
        local_feats = self.extract_local_feats(sils, contours)
        global_feats = self.extract_global_feats(sils, contours)
        return (local_feats, global_feats, pad_masks)


class HoodSlope(FeatureExtractor):

    def __init__(self, max_contour_len=512):
        self.max_contour_len = max_contour_len

    # Extract the direction vector for the next point on contour, for each given contour point.
    # The direction vector is computed by taking mean of all direction vectors in a small neighbourhood
    # around the point, so as to reduce the effect of noisy points.
    def _extract_curvature(self, ctrs, step=1, hood_delta=[-2,-1,0,1,2]):
        num_hood = len(hood_delta)
        num_ctrs = len(ctrs)
        num_sctrs = len(ctrs) // step

        feats = np.empty((num_sctrs, 2))    # direction vector for next contour point
        for idx in range(0,num_ctrs,step):
            # find neighbourhood of cpt
            hood_idxs = [((idx+delta) % num_ctrs) for delta in hood_delta]
            # extract features of each point in neighbourhood
            hood_feats = np.empty((num_hood-1, 2))
            for i in range(1,num_hood):
                diff = ctrs[hood_idxs[i]] - ctrs[hood_idxs[i-1]]
                hood_feats[i-1] = diff
            # aggregate those features
            feats[idx//step] = hood_feats.mean(axis=0)
        return feats

    def extract_sils_feat(self, sils, step=1):
        b, s, h, w = sils.shape
        feats = np.empty((b,s,self.max_contour_len,2), dtype=sils.dtype)
        pad_masks = np.empty((b,s,self.max_contour_len), dtype=bool)
        for bi in range(b):
            for si in range(s):
                cntr = self._extract_sil_contour(sils[bi][si], step=1)
                feat = self._extract_curvature(cntr, step=step)
                feats[bi][si], pad_masks[bi][si] = self._resize(feat, max_contour_len, axis=0)
        return (feats, pad_masks)
    
    def __call__(self, sils, step=2):
        feats, pad_masks = self.extract_sils_feats(sils, step=step)
        return (feats, pad_masks)

class FourierDescriptor(FeatureExtractor):
    def __init__(self, max_contour_len=512):
        self.max_contour_len = max_contour_len
    
    # Extract fourier descriptors of contours
    def _extract_fourier_descriptors(self, ctrs):
        ctrs = np.asarray(list(map(lambda pair : complex(pair[0], pair[1]), ctrs)))
        ctrs_fd = np.fft.fft(ctrs)
        ctrs_fd = np.asarray([[z.real, z.imag] for z in ctrs_fd])
        return ctrs_fd

class ShapeContext(FeatureExtractor):
    def __init__(self, max_contour_len=512):
        self.max_contour_len = max_contour_len


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CLSEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_h=2048, seq_maxlen=256):
        super(CLSEncoder, self).__init__()

        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.register_buffer('cls_pad_mask', torch.tensor([False]))
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_maxlen+1)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_h, norm_first=True, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, pad_mask=None):
        batch, seq_len, d = x.size()[:-2], x.size(-2), x.size(-1)
        assert (d == self.d_model), f"Input dimension mismatch (d_model). Expected {self.d_model} but got {d}"

        cls_tokens = self.cls_token.expand((*batch, 1, d))
        x = torch.cat((cls_tokens, x), dim=-2)
        x = x.reshape((-1, seq_len+1, d))

        if pad_mask is not None:
            pad_mask = torch.cat((self.cls_pad_mask.expand(*batch, 1), pad_mask), dim=-1)
            pad_mask = pad_mask.reshape((-1, seq_len+1))

        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)[:,0]
        x = x.reshape((*batch,d))

        return x


class RGait(BaseModel):

    def __init__(self, *args, **kargs):
        super(RGait, self).__init__(*args, **kargs)
    
    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']

        self.FeatureExtractor = HoodSlope(max_contour_len)

        self.Linear1 = nn.Linear(2, in_c[0])

        # ViT-Lite : https://arxiv.org/pdf/2104.05704.pdf
        self.SpatialEncoder = CLSEncoder(d_model=in_c[0], 
                                         num_layers=model_cfg['num_layers'], 
                                         nhead=model_cfg['num_head'], 
                                         dim_h=model_cfg['dim_h'],
                                         seq_maxlen=max_contour_len)

        self.TemporalEncoder = CLSEncoder(d_model=in_c[1],
                                          num_layers=model_cfg['num_layers'], 
                                          nhead=model_cfg['num_head'], 
                                          dim_h=model_cfg['dim_h'],
                                          seq_maxlen=max_frames_num)
        
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
        n, s, h, w = sils.size()
        del ipts

        sils = sils.detach().cpu()
        with torch.no_grad():
            feats, pad_masks = self.FeatureExtractor(sils.numpy())
            feats, pad_masks = np2t(feats, self.device), np2t(pad_masks, self.device)
        
        outs = self.Linear1(feats)
        outs = self.SpatialEncoder(outs, pad_masks)
        outs = self.TemporalEncoder(outs)

        gait = self.Head0(outs)

        if self.BN_head:    # Original Head
            bnft = self.Bn(gait)
            logi = self.Head1(bnft)
            embed = bnft
        else:               # BNNeck as Head
            bnft, logi = self.BNNecks(gait)
            embed = gait

        logi = logi.unsqueeze(2)
        embed = embed.unsqueeze(2)
        
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



        
