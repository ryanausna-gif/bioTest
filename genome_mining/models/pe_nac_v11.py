import torch
import torch.nn as nn
import torch.nn.functional as F
from .pe_nac_plus import PENACPlusDetector, pe_nac_plus_loss

class EF_PENACPlusDetector(PENACPlusDetector):
    def __init__(self,n_classes,length=1000,emb_dim=256,proj_dim=128,prototypes_per_class=4):
        super().__init__(n_classes,length,emb_dim,proj_dim,prototypes_per_class)
        self.sparse_site_head=nn.Sequential(nn.Conv1d(emb_dim,128,3,padding=1),nn.ReLU(),nn.Conv1d(128,1,1))
    def forward(self,x):
        feat=self.encoder_conv(x); h=self.pool(feat).squeeze(-1); h=self.drop(h); z=F.normalize(self.proj(h),dim=1); proto=F.normalize(self.prototypes,dim=-1)
        from .mixture_prototype import mixture_proto_logits
        mix_logits=mixture_proto_logits(z,proto,temperature=.2)
        return {'carrier_logit':self.carrier(h).squeeze(-1),'family_logits':self.family(h),'naturalness_logit':self.naturalness(h).squeeze(-1),'localization_logits':self.localizer(feat).squeeze(1),'sparse_site_logits':self.sparse_site_head(feat).squeeze(1),'embedding':h,'projection':z,'mixture_proto_logits':mix_logits}

def ef_pe_nac_loss(out,y,family_labels,naturalness,loc_mask,**kwargs):
    base=pe_nac_plus_loss(out,y,family_labels,naturalness,loc_mask,**kwargs)
    if 'sparse_site_logits' in out and loc_mask is not None:
        base=base+.1*F.binary_cross_entropy_with_logits(out['sparse_site_logits'],loc_mask,pos_weight=torch.tensor(4.0,device=loc_mask.device))
    return base
