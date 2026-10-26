# original https://github.com/mlfoundations/open_clip (not for this code)
from src import clip # super important line as we are replaceing older open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F
from geom_hyp import lorentz as L
from geom_hyp import manifold_conversion as MC



class OutputDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    
    def __setattr__(self, name, value):
        self[name] = value

def prototype_similarity_loss(_d, label, tau=2.0):
    # Squared exponential similarity loss:
    # gradient vanishes for large prototype distances
    """  _d:   [B, C] prototype distances
    label: [B] class indices """
    d_pos = _d[torch.arange(_d.size(0), device=_d.device), label]
    sim_pos = torch.exp(-d_pos / tau)
    return ((1.0 - sim_pos) ** 2).mean()


def smooth_margin_loss(_d, label, margin = 0.3, tau=0.1):
    # Smooth approximation of max(0, d - margin)
    d_pos = _d[torch.arange(_d.size(0), device=_d.device),label]
    loss = tau * F.softplus((d_pos - margin) / tau)
    return loss.mean() # (loss**2).mean() for higher gradient at distant

def prototype_margin_loss(_d, label, margin = 0.3):
    # Exact hinge margin: max(0, d - margin)
    """_d:    [B, C] prototype distances
    label: [B] class indices"""
    d_pos = _d[ torch.arange(_d.size(0), device=_d.device),label]
    loss =  F.relu(d_pos - margin)
    return loss.mean()

class MonetWithHead(nn.Module):
    def __init__(self, num_classes = 11, loss_type="ce", focal_gamma=2.0,
                        multilayer=False, hidden_dim=256, clip_proj = False):
        super().__init__()
        model = clip.load("ViT-L/14", jit=False)[0]
        model.load_state_dict(torch.hub.load_state_dict_from_url("https://aimslab.cs.washington.edu/MONET/weight_clip.pt",map_location="cpu"))
        model = model.float()
        self.visual = model.visual
        if clip_proj:
            # Keep CLIP's original projection
            embed_dim = self.visual.proj.shape[1]
        else:
            # Remove CLIP projection
            self.visual.proj = None
            # pre-projection representation after ln_post
            embed_dim = self.visual.ln_post.normalized_shape[0]
        
        
        if num_classes > 0:
            if multilayer:
                self.clf_head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes),)
                # near zero initialization
                nn.init.normal_(self.clf_head[-1].weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.clf_head[-1].bias)

            else:
                self.clf_head = nn.Linear(embed_dim, num_classes)
                # near zero initialization
                nn.init.normal_(self.clf_head.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.clf_head.bias)
        else:
            self.clf_head = nn.Identity() 
            
        loss_type = loss_type.lower()
        if loss_type not in {"ce", "focal"}:
            raise ValueError(f"loss_type must be 'ce' or 'focal', got {loss_type!r}.")
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
    
    def Get_CE_loss(self, out, label, ignore_index=-100):
        """
        Args:
            out:   logits [B, C]
            label: class indices [B] or one-hot labels [B, C]
        """
        if label.ndim == 1:
            label = label.to(device=out.device, dtype=torch.long)
            return F.cross_entropy(out,label,ignore_index=ignore_index,reduction="mean",)
    
        elif label.ndim == 2:
            if label.shape != out.shape:
                raise ValueError(
                    f"One-hot labels must match logits shape: "
                    f"{label.shape} versus {out.shape}.")
    
            label = label.to(device=out.device, dtype=out.dtype)
            # Equivalent to ignore_index for one-hot targets
            if ignore_index >= 0:
                keep = label.argmax(dim=1) != ignore_index
                out = out[keep]
                label = label[keep]
    
            return F.cross_entropy(out, label,  reduction="mean",)
    
        else:
            raise ValueError(f"label must have shape [B] or [B, C], got {label.shape}.")

    def Get_Focal_loss(self, out, label, gamma=2.0, alpha=1.0):
        log_prob = F.log_softmax(out, dim=1)
        prob = log_prob.exp()  
        if label.ndim == 1:
            label = label.to(out.device, dtype=torch.long)
    
            log_pt = log_prob.gather(1, label[:, None]).squeeze(1)
            pt = log_pt.exp()
            loss = -alpha * (1.0 - pt).pow(gamma) * log_pt
    
        elif label.ndim == 2:
            if label.shape != out.shape:
                raise ValueError(
                    f"Label shape {label.shape} must match {out.shape}." )
            label = label.to(out.device, dtype=out.dtype)
            loss = -alpha * (  label * (1.0 - prob).pow(gamma) * log_prob).sum(dim=1)
    
        else: raise ValueError(f"label must be [B] or [B, C], got {label.shape}." )
    
        return loss.mean()

    def Get_classification_loss(self, out, label, ignore_index = -100):
        if self.loss_type == "ce":
            return self.Get_CE_loss(out, label, ignore_index)
        if self.loss_type == "focal":
            return self.Get_Focal_loss(
                out,
                label,
                gamma=self.focal_gamma,)
        raise RuntimeError(f"Unsupported loss type: {self.loss_type!r}.")



    def forward(self, x, label =  None):
        features = self.visual(x)
        out = self.clf_head(features)
        
        if label is None:
            loss = None
        else:
            # label need to be one-hot or argmax()
            loss = self.Get_classification_loss(out, label)
        
        out_dict = OutputDict({'output' : out, 
                             'loss': loss})
        return out_dict



def clip_safe(input_vector, r = 2.3):
    norms = torch.norm(input_vector, dim=-1, keepdim=True).clamp(min=1e-8)
    return input_vector * (float(r) / norms).clamp(max=1)



class MonetWithHypHead(MonetWithHead):
    def __init__(
        self, embed_dim = 32, l_target = None, loss_type = 'ce',
        curv_init: float = 1.0,
        learn_curv: bool = False,
        clip_value = 3.8, # clip value  = 6.0 #  minimum of max(L.log_map0(model.l_target)).norm(2,-1)) to match the velocity.
        entail_weight = 0.2, 
         focal_gamma=2.0,
        logit_scale = 1.0, 
        multilayer=False,
        hidden_dim=256,
    ):
        
        super().__init__(num_classes=embed_dim, loss_type=  loss_type, focal_gamma = focal_gamma, 
                         multilayer = multilayer, hidden_dim= hidden_dim)
        '''
        l_target: lorentz point for the target: size of [class_time, space_dims (embed_dim = 32)]
        '''
        self.l_target = nn.Parameter(l_target.float(), requires_grad=False) # should of the same dimension of embed_dim 
        # self.register_buffer("l_target", l_target) # alternative to earler command
        self.visual_alpha =  nn.Parameter(torch.tensor(1.0/2.0).log())
        self.curv = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale).log(), requires_grad = False) # temperature parameter # 2 looks good at the beginning
        self.clip_value = clip_value
        # self.register_buffer("clip_value", torch.tensor(clip_value))
        self.entailment_weight = entail_weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
        
    def encode_image(self, images: torch.Tensor, project : bool = True,  mul_factor = 1.0):
        image_feats = self.visual(images)
        image_feats = self.clf_head(image_feats) # projection head
                                   
        if project:
            # scaling (do we need it)
            image_feats = image_feats * self.visual_alpha.exp() * mul_factor
            # clipping hyperbolic tangent vector
            image_feats = clip_safe(image_feats, self.clip_value) 
            # exponential mapping
            with torch.autocast(device_type=image_feats.device.type, enabled=False):
                image_feats = L.exp_map0(image_feats.float(),  self.curv.exp().float())
        return image_feats
        
    def forward(self, x, label=None, ignore_index = -100,  mul_factor = 1.0):
        '''
        label: class indexes
        '''
        with torch.no_grad():
            self.visual_alpha.clamp_(max=0.0)
            self.logit_scale.clamp_(max=4.6052)
        image_feats = self.encode_image(x, project = True, mul_factor = mul_factor)
        # text_feats = self.l_target

        if label is None:
            loss, classification_loss, entailment_loss = None, None, None
        else:
            label = label.to(device=image_feats.device,dtype=torch.long)

            keep = label != ignore_index # ignore any class if required
            image_feats = image_feats[keep]
            label = label[keep]

            with torch.autocast(device_type=image_feats.device.type,enabled=False,):
                _a, _d, _ah = L.oxy_angle_dist( image_feats.float(),self.l_target.float(),label, min_radius = 0.2,)
                logits = -self.logit_scale.exp().float() * _d
                entailment_loss = F.relu(_a - _ah).mean()

            classification_loss  = self.Get_classification_loss( logits, label) # for focal loss or cross entropy loss 
            
            prototype_loss = smooth_margin_loss(_d, label, margin= 0.4, tau = 0.25) #prototype_similarity_loss(_d, label, tau = 0.5)
            loss = classification_loss + 0.2 * prototype_loss # additional loss

            # loss = smooth_margin_loss(_d, label, margin= 0.5, tau = 0.25)
            
            if self.entailment_weight > 0:
                loss = loss + self.entailment_weight * entailment_loss # additional loss
                    
        
        out_dict = OutputDict({
            'output': image_feats, # lorentz spatial co ordinates
            'loss': loss, 
            'classification_loss': classification_loss,
            'entailment_loss': entailment_loss, 
        })
        return out_dict   



# cpcc: implicit structure learning 