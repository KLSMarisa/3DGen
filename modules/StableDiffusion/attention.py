from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import numpy as np
from .util import checkpoint, timestep_embedding


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

try:
    import flash_attn
    FLASHATTEN_IS_AVAILBLE = True
except:
    FLASHATTEN_IS_AVAILBLE = True

# FLASHATTEN_IS_AVAILBLE = False
# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def Normalize_tri(in_channels):
    return torch.nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)

class CrossAttention_triplane(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print(q.shape, k.shape, v.shape, self.heads)
        # print(self.scale)

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
        # print(sim.shape)
        # print(sim.dtype)
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', sim, v.float())
        # print('out',out.shape)
        # exit()
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #       f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        # print("innerdim", inner_dim)
        # print("context",context_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        # x = rearrange(x, 'b (h w) c -> b h w c', h=64).contiguous()
        q = self.to_q(x)
        context = default(context, x)
        # print(context)
        # print("shape:",context.shape)
        k = self.to_k(context)
        v = self.to_v(context)
        # print(x.shape)
        b, _, _ = q.shape
        # print(q.shape, k.shape, v.shape, self.heads)

        q = rearrange(q, 'b l (h c) -> b l h c', h=self.heads)
        k = rearrange(k, 'b l (h c) -> b l h c', h=self.heads)
        v = rearrange(v, 'b l (h c) -> b l h c', h=self.heads)


        # ## plan1 H->head  c->D
        # pass
        #
        # ## plan2 h->h
        # q = rearrange(q, 'b d w (h c) -> b w h (c d)', h=self.heads)
        # k = rearrange(k, 'b d w (h c) -> b w h (c d)', h=self.heads)
        # v = rearrange(v, 'b d w (h c) -> b w h (c d)', h=self.heads)
        # print(q.shape, k.shape, v.shape, self.heads)
        # exit()
        # actually compute the attention, what we cannot get enough of
        
        if FLASHATTEN_IS_AVAILBLE:
            # print('flash')
            out = flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
            # print(out)

        elif XFORMERS_IS_AVAILBLE:
            # print('xfo')
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        
        # with torch.backends.cuda.enable_flash_sdp():
        #     out = F.scaled_dot_product_attention(q, k, v, scale=None, is_causal=False)
        # print(out.shape)
        # exit()
        if exists(mask):
            raise NotImplementedError
        out = rearrange(out, 'b l h c -> b l (h c)')

        return self.to_out(out)


class MemoryEfficientCrossAttention_tri(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #       f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        # print("innerdim", inner_dim)
        # print("context",context_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        # print(x.shape)
        q = self.to_q(x)
        if context==None:
            # print('tri_atten')
            # print(q.shape)
            q = rearrange(q, '(b t) l c -> b t l c', t=3)
            
            # print(q.shape)
            # exit()
            ## triplane split
            plane1 = q[:,0,:,:].clone()
            plane2 = q[:,1,:,:].clone()
            plane3 = q[:,2,:,:].clone()
            # context = default(context, x)
            # print(context)
            # print("shape:",context.shape)
            # k = self.to_k(context)
            # v = self.to_v(context)
            # print('plane:',plane1.shape)
            b, l, d = plane1.shape ## B,4096, C*H
            # print(q.shape, k.shape, v.shape, self.heads)

            

            h=w = int(np.sqrt(l))
            feat1 = plane1.clone().reshape(b,h,w,d)
            feat2 = plane2.clone().reshape(b,h,w,d)
            feat3 = plane3.clone().reshape(b,h,w,d)


            i = torch.arange(w)
            j = torch.arange(h)
            yy, xx = torch.meshgrid(i, j)
            i=xx.reshape(h*w)
            j=yy.reshape(h*w)

            # print(i[:32])
            # print(j[:32])
            # print(h)
            # exit()
            ## plane1 as q
            feat = torch.cat((feat2[:, :, i, :], feat2[:, int(h/2 - 1), :, :].unsqueeze(2).repeat(1,1,l,1)),dim=1)
            p2_1 = feat.permute(0,2,1,3)
            feat = torch.cat((feat3[:, :, int(w/2 - 1), :].unsqueeze(2).repeat(1,1,l,1).permute(0,2,1,3), feat3[:, j, :, :]),dim=2)
            p3_1 = feat
            
            ## plane2 as q
            feat = torch.cat((feat1[:, :, i, :], feat1[:, int(h/2 - 1), :, :].unsqueeze(2).repeat(1,1,l,1)),dim=1)
            p1_2 = feat.permute(0,2,1,3)
            feat = torch.cat((feat3[:, :, w-1-j, :], feat3[:, int(h/2 - 1), :, :].unsqueeze(2).repeat(1,1,l,1)),dim=1)
            p3_2 = feat.permute(0,2,1,3)



            ## plane3 as q
            feat = torch.cat((feat1[:, :, int(w/2 - 1), :].unsqueeze(2).repeat(1,1,l,1).permute(0,2,1,3), feat1[:, j, :, :]),dim=2)
            p1_3 = feat
            feat = torch.cat((feat3[:, :, int(w/2 - 1), :].unsqueeze(2).repeat(1,1,l,1).permute(0,2,1,3), feat3[:, h-1-i, :, :]),dim=2)
            p2_3 = feat


            p1_23 = torch.cat((p2_1, p3_1), dim=2)
            p2_13 = torch.cat((p1_2, p3_2), dim=2)
            p3_12 = torch.cat((p1_3, p2_3), dim=2)

            q = torch.cat((plane1, plane2, plane3), dim=0).unsqueeze(2) # 3b l 1 (hc) 
            k = torch.cat((p1_23, p2_13, p3_12), dim=0)
            v = k

            # del p1_23
            # del p2_13
            # del p3_12
            # del feat1
            # del feat2
            # del feat3
            # torch.cuda.empty_cache()
            
            # exit()
            q = rearrange(q, 'b l f (h c) -> (b l) f h c', h=self.heads)
            k = rearrange(k, 'b l f (h c) -> (b l) f h c', h=self.heads)
            v = rearrange(v, 'b l f (h c) -> (b l) f h c', h=self.heads)
            q_chunks = torch.chunk(q, 64, dim=0)
            k_chunks = torch.chunk(k, 64, dim=0)
            v_chunks = torch.chunk(v, 64, dim=0)
            # print(q.shape, k_chunks[0].shape, v.shape)
            # exit()
            # actually compute the attention, what we cannot get enough of
        
            results = []
            for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
                if FLASHATTEN_IS_AVAILBLE:
                    # print('flash')
                    # out = flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
                    out = flash_attn.flash_attn_func(q_chunk, k_chunk, v_chunk, dropout_p=0.0, softmax_scale=None, causal=False)
                    # print(out.shape)
                    results.append(out)
                    # xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
                    # print(out)

                elif XFORMERS_IS_AVAILBLE:
                    # print('xfo')
                    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
            out = torch.cat(results, dim=0)
            # print(out.shape)
            # exit()
            # with torch.backends.cuda.enable_flash_sdp():
            #     out = F.scaled_dot_product_attention(q, k, v, scale=None, is_causal=False)
            
            if exists(mask):
                raise NotImplementedError
            out = rearrange(out, '(b l) f h c -> b (l f) (h c)', l=l)
        else:
            # print('cross_atten')
            context = default(context, x)
            # print(context)
            # print("shape:",context.shape)
            k = self.to_k(context)
            v = self.to_v(context)
            # print(x.shape)
            b, _, _ = q.shape
            # print(q.shape, k.shape, v.shape, self.heads)

            q = rearrange(q, 'b l (h c) -> b l h c', h=self.heads)
            k = rearrange(k, 'b l (h c) -> b l h c', h=self.heads)
            v = rearrange(v, 'b l (h c) -> b l h c', h=self.heads)


            # ## plan1 H->head  c->D
            # pass
            #
            # ## plan2 h->h
            # q = rearrange(q, 'b d w (h c) -> b w h (c d)', h=self.heads)
            # k = rearrange(k, 'b d w (h c) -> b w h (c d)', h=self.heads)
            # v = rearrange(v, 'b d w (h c) -> b w h (c d)', h=self.heads)
            # print(q.shape, k.shape, v.shape, self.heads)
            # exit()
            # actually compute the attention, what we cannot get enough of
            
            if FLASHATTEN_IS_AVAILBLE:
                # print('flash')
                out = flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
                # print(out)

            elif XFORMERS_IS_AVAILBLE:
                # print('xfo')
                out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        
            # with torch.backends.cuda.enable_flash_sdp():
            #     out = F.scaled_dot_product_attention(q, k, v, scale=None, is_causal=False)
            # print(out.shape)
            # exit()
            if exists(mask):
                raise NotImplementedError
            out = rearrange(out, 'b l h c -> b l (h c)')
        # print('attn_out',out.shape)
        # exit()
        

        # ## 将 W维度求和，得到 b h 1 D, 再复制至 b h w D
        # out = torch.sum(out, dim=2, keepdim=True)
        # print(out.shape)
        # out = out.expand(-1, -1, 64, -1)
        # print(out.shape)
        # out = rearrange(out, 'b h w D -> b (h w) D', h=64)
        # ### h
        # x = out + x
        # q = self.to_q(x)
        # context = default(context, x)
        # # print(context)
        # # print("shape:",context.shape)
        # k = self.to_k(context)
        # v = self.to_v(context)
        # print(x.shape)
        # b, _, _ = q.shape
        # print(q.shape, k.shape, v.shape, self.heads)

        # q = rearrange(q, 'b l (h c) -> b l h c', h=self.heads)
        # k = rearrange(k, 'b l (h c) -> b l h c', h=self.heads)
        # v = rearrange(v, 'b l (h c) -> b l h c', h=self.heads)

        # # ## plan1 H->head  c->D
        # # pass
        # #
        # # ## plan2 h->h
        # # q = rearrange(q, 'b d w (h c) -> b w h (c d)', h=self.heads)
        # # k = rearrange(k, 'b d w (h c) -> b w h (c d)', h=self.heads)
        # # v = rearrange(v, 'b d w (h c) -> b w h (c d)', h=self.heads)
        # print(q.shape, k.shape, v.shape, self.heads)

        # # actually compute the attention, what we cannot get enough of
        # if FLASHATTEN_IS_AVAILBLE:
        #     # print('flash')
        #     out = flash_attn.flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        #     # print(out)

        # elif XFORMERS_IS_AVAILBLE:
        #     # print('xfo')
        #     out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # if exists(mask):
        #     raise NotImplementedError
        # out = rearrange(out, 'b l h c -> b l (h c)')
        # print(out.shape)
        # out = rearrange(out, 'b (h w) D -> b h w D', h=64)

        # ## 将 H 维度求和，得到 b 1 w D, 再复制至 b h w D
        # out = torch.sum(out, dim=1, keepdim=True)
        # print(out.shape)
        # out = out.expand(-1, 64, -1, -1)
        # print(out.shape)
        # out = rearrange(out, 'b h w D -> b (h w) D', h=64)
        # exit()
        # with open('//home/caixiao/projects/3DGen/test_out/test_out.txt', 'a') as f:
        #     f.write(f'attn1 flash before  \n {out} \n')
        # exit()
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-exlib": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-exlib" if XFORMERS_IS_AVAILBLE or FLASHATTEN_IS_AVAILBLE else "softmax"
        # attn_mode = "softmax"
        # print(attn_mode)
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # print('context_dim',context_dim)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x

        ## tri
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerBlock_tri(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-exlib": MemoryEfficientCrossAttention_tri
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-exlib" if XFORMERS_IS_AVAILBLE or FLASHATTEN_IS_AVAILBLE else "softmax"
        # attn_mode = "softmax"
        # print(attn_mode)
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # print('context_dim',context_dim)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        # x = self.attn2(self.norm2(x), context=context) + x
        # x = self.ff(self.norm3(x)) + x

        ## tri
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        self.context_dim = context_dim
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_head = d_head
        inner_dim = n_heads * d_head
        self.depth = depth
        self.dropout = dropout
        self.disable_self_attn = disable_self_attn
        self.use_checkpoint = use_checkpoint
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        # print('x',x.shape)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x_in = x
        # print('x_in',x.shape)
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # x = rearrange(x, 'b c h w -> b h w c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            ctx = repeat(context[i], 'b l c -> (b f) l c', f=f)
            x = block(x, context=ctx)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        out = rearrange(out, '(b f) c h w -> b c f h w', b=b)
        return out


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_head = d_head
        inner_dim = n_heads * d_head
        self.depth = depth
        self.dropout = dropout
        self.disable_self_attn = disable_self_attn
        self.use_checkpoint = use_checkpoint
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b h w) c f')
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c f -> b f c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        t = torch.arange(0, x.shape[1], 1, dtype=torch.int64, device=x.device)
        t_emb = timestep_embedding(t, x.shape[2], repeat_only=False)
        x = x + t_emb.to(x.dtype).unsqueeze(0)
        for i, block in enumerate(self.transformer_blocks):
            ctx = repeat(context[i], 'b l c -> (b h w) l c', h=h, w=w)
            x = block(x, context=ctx)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b f c -> b c f').contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        out = rearrange(out, '(b h w) c f -> b c f h w', b=b, h=h, w=w)
        return out


class TriplaneTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        self.context_dim = context_dim
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_head = d_head
        inner_dim = n_heads * d_head
        self.depth = depth
        self.dropout = dropout
        self.disable_self_attn = disable_self_attn
        self.use_checkpoint = use_checkpoint
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        # if inner_dim<960:
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock_tri(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        # else:
        #     self.transformer_blocks = nn.ModuleList(
        #         [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
        #                             disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
        #             for d in range(depth)]
        #     )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        # print('x',x.shape)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x_in = x
        # print('x_in',x.shape)
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # x = rearrange(x, 'b c h w -> b h w c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        # print(context)
        if context == []:
            for i, block in enumerate(self.transformer_blocks):
            # ctx = repeat(context[i], 'b l c -> (b f) l c', f=f)
                x = block(x)
                # print(x.shape)
        else:
            for i, block in enumerate(self.transformer_blocks):
                ctx = repeat(context[i], 'b l c -> (b f) l c', f=f)
                x = block(x,context=ctx)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        
        out = x + x_in
        # print("out",out.shape)
        out = rearrange(out, '(b f) c h w -> b c f h w', b=b)
        return out