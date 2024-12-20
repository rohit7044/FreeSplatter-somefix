import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange, repeat
import xformers.ops as xops


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        out = xops.memory_efficient_attention(q, k, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()

        self.self_attn = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4, bias=False),
            nn.GELU(),
            nn.Linear(dim*4, dim, bias=False),
        )

        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.norm2 = nn.LayerNorm(dim, bias=False)

    def forward(self, x, context=None):
        before_sa = self.norm1(x)
        x = x + self.self_attn(before_sa)
        x = self.ff(self.norm2(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self, 
        image_size=512, 
        patch_size=8, 
        input_dim=3, 
        inner_dim=1024,
        output_dim=14,
        n_heads=16, 
        depth=24, 
        dropout=0.,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim

        self.patchify = nn.Conv2d(input_dim, inner_dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, inner_dim))
        self.ref_embed = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.src_embed = nn.Parameter(torch.zeros(1, 1, inner_dim))

        self.blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, inner_dim//n_heads, dropout=dropout)
                for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(inner_dim, bias=False)
        self.unpatchify = nn.Linear(inner_dim, patch_size ** 2 * output_dim, bias=True)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.ref_embed, std=.02)
        nn.init.trunc_normal_(self.src_embed, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[-2]
        N = self.pos_embed.shape[-2]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2).contiguous(),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim).contiguous()
        return patch_pos_embed

    def forward(self, images):
        """
        images: (B, N, C, H, W)
        """
        B, N, _, H, W = images.shape

        # patchify
        images = rearrange(images, 'b n c h w -> (b n) c h w')
        tokens = self.patchify(images)
        tokens = rearrange(tokens, 'bn c h w -> bn (h w) c')

        # add pos encodings
        tokens = rearrange(tokens, '(b n) hw c -> b n hw c', b=B)
        tokens = tokens + self.interpolate_pos_encoding(tokens, W, H).unsqueeze(1)
        view_embeds = torch.cat([self.ref_embed, self.src_embed.repeat(1, N-1, 1)], dim=1)
        tokens = tokens + view_embeds.unsqueeze(2)

        # tokens = rearrange(tokens, '(b n) hw c -> b n hw c', b=B)
        # tokens = tokens + self.interpolate_pos_encoding(tokens, W, H).unsqueeze(1)
        # view_embeds = self.src_embed.repeat(1, N, 1)
        # view_embeds[:, 0:1] = torch.zeros_like(self.ref_embed)
        # tokens = tokens + view_embeds.unsqueeze(2)

        # transformer
        tokens = rearrange(tokens, 'b n hw c -> b (n hw) c')
        x = tokens
        for layer in self.blocks:
            x = layer(x)
        
        # unpatchify
        x = self.norm(x)
        x = self.unpatchify(x)
        x = rearrange(x, 'b (n h w) c -> b n h w c', n=N, h=H//self.patch_size, w=W//self.patch_size)
        x = rearrange(x, 'b n h w (p q c) -> b n (h p) (w q) c', p=self.patch_size, q=self.patch_size)
        out = x

        return out
