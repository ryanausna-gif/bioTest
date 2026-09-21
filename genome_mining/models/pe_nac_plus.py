import torch
import torch.nn as nn
import torch.nn.functional as F
from .mixture_prototype import mixture_proto_logits
from .hierarchical_contrastive import supervised_contrastive_loss as supcon_loss

class PENACPlusDetector(nn.Module):
    def __init__(self, n_classes, length=1000, emb_dim=256, proj_dim=128, prototypes_per_class=4):
        super().__init__(); self.length=int(length); self.n_classes=int(n_classes); self.prototypes_per_class=int(prototypes_per_class)
        self.encoder_conv=nn.Sequential(nn.Conv1d(4,64,9,padding=4),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,128,11,padding=5),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,emb_dim,13,padding=6),nn.BatchNorm1d(emb_dim),nn.ReLU())
        self.pool=nn.AdaptiveMaxPool1d(1); self.drop=nn.Dropout(0.25)
        self.carrier=nn.Linear(emb_dim,1); self.family=nn.Linear(emb_dim,n_classes); self.naturalness=nn.Linear(emb_dim,1)
        self.proj=nn.Sequential(nn.Linear(emb_dim,emb_dim),nn.ReLU(),nn.Linear(emb_dim,proj_dim))
        self.localizer=nn.Sequential(nn.Conv1d(emb_dim,128,5,padding=2),nn.ReLU(),nn.Conv1d(128,1,1))
        self.prototypes=nn.Parameter(torch.randn(n_classes,prototypes_per_class,proj_dim)*0.02)
    def forward(self,x):
        feat=self.encoder_conv(x); h=self.drop(self.pool(feat).squeeze(-1)); z=F.normalize(self.proj(h),dim=1); proto=F.normalize(self.prototypes,dim=-1)
        return {'carrier_logit':self.carrier(h).squeeze(-1),'family_logits':self.family(h),'naturalness_logit':self.naturalness(h).squeeze(-1),'localization_logits':self.localizer(feat).squeeze(1),'embedding':h,'projection':z,'mixture_proto_logits':mixture_proto_logits(z,proto,temperature=0.2)}

def pe_nac_plus_loss(out,y,family_labels,naturalness,loc_mask,family_weight=0.35,naturalness_weight=0.3,contrast_weight=0.1,prototype_weight=0.3,localization_weight=0.25,episodic_weight=0.2,pseudo_unknown_mask=None):
    loss=F.binary_cross_entropy_with_logits(out['carrier_logit'],y)
    loss=loss+family_weight*F.cross_entropy(out['family_logits'],family_labels)
    loss=loss+naturalness_weight*F.binary_cross_entropy_with_logits(out['naturalness_logit'],naturalness)
    if contrast_weight>0: loss=loss+contrast_weight*supcon_loss(out['projection'],family_labels)
    if prototype_weight>0: loss=loss+prototype_weight*F.cross_entropy(out['mixture_proto_logits'],family_labels)
    if localization_weight>0 and loc_mask is not None:
        loss=loss+localization_weight*F.binary_cross_entropy_with_logits(out['localization_logits'],loc_mask,pos_weight=torch.tensor(4.0,device=loc_mask.device))
    if episodic_weight>0 and pseudo_unknown_mask is not None and pseudo_unknown_mask.any():
        prob=F.softmax(out['mixture_proto_logits'],dim=1); loss=loss+episodic_weight*prob.max(dim=1).values[pseudo_unknown_mask].mean()
    return loss
