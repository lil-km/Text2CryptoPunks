from text2punks.attention import Attention, SparseAxialCausalAttention

from torch import nn

from functools import partial
from itertools import islice, cycle


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, depth = 1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class SequentialSequence(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for (f, g) in list(self.layers):
            x = x + f(x)
            x = x + g(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        causal = True,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        resid_dropout = 0.,
        embd_dropout = 0.,
        ff_dropout = 0.,
        image_size = 24,
        attn_types = None,
    ):
        super().__init__()
        layers = nn.ModuleList([])

        attn_types = default(attn_types, ("full",))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for attn_type in attn_type_layer:
            if attn_type == "full":
                attn_class = partial(Attention, causal = causal)
            elif attn_type == "axial_row":
                attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 0, image_size = image_size)
            elif attn_type == "axial_col":
                attn_class = partial(SparseAxialCausalAttention, seq_len = seq_len, axis = 1, image_size = image_size)
            else:
                raise ValueError(f"attention type "{attn_type}" is not valid")

            attn = attn_class(dim, seq_len = seq_len, heads = heads, dim_head = dim_head, attn_dropout = attn_dropout, resid_dropout = resid_dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
            ]))

        # full attention in the last layer

        attn_class = partial(Attention, causal = causal)
        attn = attn_class(dim, seq_len = seq_len, heads = heads, dim_head = dim_head, attn_dropout = attn_dropout, resid_dropout = resid_dropout)

        layers.append(nn.ModuleList([
            PreNorm(dim, attn),
            PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
        ]))

        self.layers = SequentialSequence(layers)
        self.embd_drop = nn.Dropout(embd_dropout)

    def forward(self, x):
        x = self.embd_drop(x)
        return self.layers(x)
        