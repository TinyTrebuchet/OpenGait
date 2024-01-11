import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from einops.layers.torch import Rearrange, Reduce

from ..base_model import BaseModel
from ..modules import PackSequenceWrapper, SeparateBNNecks

def compute_fdei(seqs):
    sz = seqs.size()[2]
    fdei = [torch.zeros(seqs[:,:,0].size()).cuda()]
    for t in range(1,sz):
        fwd_fdei = seqs[:,:,t] - seqs[:,:,t-1]
        fwd_fdei = F.relu(fwd_fdei)
        bck_fdei = seqs[:,:,t-1] - seqs[:,:,t]
        bck_fdei = F.relu(bck_fdei)
        fdei.append(fwd_fdei + bck_fdei)
    fdei = torch.stack(fdei).permute((1,2,0,3,4))
    return fdei

class FactorizedEncoder(nn.Module):
    def __init__(self, input_dims, patch_dims, dim=192, nhead=3, ffn_dim=192*3, out_dim=128, layers=4, in_channels=1, emb_dropout=0.):
        super(FactorizedEncoder, self).__init__()

        assert len(input_dims) == 3 and len(patch_dims) == 3, "Input & Patch dimension should be (t h w)"
        assert np.sum(np.remainder(input_dims, patch_dims)) == 0, "Frames cannot be divided into patches"

        nt, nh, nw = np.floor_divide(input_dims, patch_dims)
        patch_dim = in_channels * np.prod(patch_dims)
        self.embed_to_patches = nn.Sequential(
            Rearrange('b c (nt p1) (nh p2) (nw p3) -> b (nh nw) nt (p1 p2 p3 c)', p1=patch_dims[0], p2=patch_dims[1], p3=patch_dims[2]),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, nh*nw, nt+1, dim))
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.spatial_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=ffn_dim, batch_first=True, norm_first=True),
            num_layers=layers
        )
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=ffn_dim, batch_first=True, norm_first=True),
            num_layers=layers
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x):
        print(x.size())
        x = self.embed_to_patches(x)
        print(x.size())
        print(self.pos_embedding.size())
        b, n, t, _ = x.shape

        temporal_cls_tokens = repeat(self.temporal_cls_token, '() t d -> b n t d', b=b, n=n)
        x = torch.cat((temporal_cls_tokens, x), dim=2)
        x += self.pos_embedding[:,:,:(t+1)]
        x = self.dropout(x)

        x = rearrange(x, 'b n t d -> (b n) t d')
        x = self.temporal_encoder(x)
        x = rearrange(x[:,0], '(b n) ... -> b n ...', b=b)

        spatial_cls_tokens = repeat(self.spatial_cls_token, '() n d -> b n d', b=b)
        x = torch.cat((spatial_cls_tokens, x), dim=1)

        x = self.spatial_encoder(x)[:, 0]
        x = self.mlp_head(x)

        return x


class FDFormer(BaseModel):

    def __init__(self, *args, **kargs):
        super(FDFormer, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        num_frames = model_cfg['frames_num']
        img_size = model_cfg['image_size']
        input_dims = (num_frames, *img_size)

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, in_c[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.LTA = nn.Sequential(
            nn.Conv3d(in_c[0], in_c[0], kernel_size=(3,1,1), stride=(3,1,1), padding=(0,0,0)),
            nn.LeakyReLU(inplace=True)
        )

        self.GConv = nn.Sequential(
            nn.Conv3d(in_c[0], in_c[1], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_c[1], in_c[2], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.TP = PackSequenceWrapper(torch.max)
        self.SP = Reduce('b c h w -> b c', 'max')

        self.FAEN = FactorizedEncoder(input_dims, patch_dims=(4,4,4))

        self.Head = nn.Sequential(
            nn.LayerNorm(2*in_c[-1]),
            nn.Linear(2*in_c[-1], in_c[-1])
        )
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
    
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0].unsqueeze(1)
        del ipts

        outs1 = self.conv3d(sils)                                       # [b c s h w] -> [64, 32, 30, 64, 44]
        outs1 = self.LTA(outs1)                                         # [b c s h w] -> [64, 32, 10, 64, 44]
        outs1 = self.GConv(outs1)                                       # [b c s h w] -> [64, 128, 10, 64, 44]
        outs1 = self.TP(outs1, seqL=seqL, options={"dim": 2})[0]        # [b c h w]   -> [64, 128, 64, 44]
        outs1 = self.SP(outs1)                                          # [b c]       -> [64, 128]

        outs2 = compute_fdei(sils)
        outs2 = self.FAEN(outs2)

        outs = torch.cat((outs1, outs2), dim=1)
        gait = self.Head(outs)

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
