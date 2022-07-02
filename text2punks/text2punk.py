from axial_positional_embedding import AxialPositionalEmbedding
from text2punks.transformer import Transformer

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

import math


# helpers fns

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers fn

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

# main CLIP class

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 10000,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        num_visual_tokens = 256,
        visual_enc_depth = 6,
        visual_image_seq_len = 256,
        visual_image_size = 24,
        visual_heads = 8,
        attn_pdrop = 0.1,
        resid_pdrop = 0.1,
        embd_pdrop = 0.1,
        ff_dropout = 0.1,
        attn_types = None
    ):
        super().__init__()

        # Texts

        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)

        self.text_transformer = Transformer(
            dim = dim_text,
            causal = False,
            seq_len = text_seq_len,
            depth = text_enc_depth,
            heads = text_heads,
            dim_head = dim_text // text_heads,
            attn_dropout = attn_pdrop,
            resid_dropout = resid_pdrop,
            embd_dropout = embd_pdrop,
            ff_dropout = ff_dropout,
            attn_types = attn_types
        )

        self.text_ln = nn.LayerNorm(dim_text)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # Images

        self.image_emb = nn.Embedding(num_visual_tokens, dim_image)
        self.image_pos_emb = nn.Embedding(visual_image_seq_len, dim_image)

        self.visual_transformer = Transformer(
            dim = dim_image,
            causal = False,
            seq_len = visual_image_seq_len,
            depth = visual_enc_depth,
            heads = visual_heads,
            dim_head = dim_image // visual_heads,
            attn_dropout = attn_pdrop,
            resid_dropout = resid_pdrop,
            embd_dropout = embd_pdrop,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_size = visual_image_size,
        )

        self.image_ln = nn.LayerNorm(dim_image)
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        self.temperature = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        text,
        image,
        return_loss = False
    ):
        b, device= text.shape[0], text.device

        text_emb = self.text_emb(text)
        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        image_emb = self.image_emb(image)
        image_emb += self.image_pos_emb(torch.arange(image.shape[1], device = device))

        enc_text = self.text_transformer(text_emb)
        enc_image = self.visual_transformer(image_emb)

        text_latents = enc_text.mean(dim = 1)
        image_latents = enc_image.mean(dim = 1)

        text_latents = self.text_ln(text_latents)
        image_latents = self.image_ln(image_latents)

        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        text_latents, image_latents = map(lambda t: F.normalize(t, p = 2, dim = -1), (text_latents, image_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum("n d, n d -> n", text_latents, image_latents) * temp
            return sim

        sim = einsum("i d, j d -> i j", text_latents, image_latents) * temp
        labels = torch.arange(b, device = device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss

# main Text2Punks class

class Text2Punks(nn.Module):
    def __init__(
        self,
        *,
        n_embd,
        n_layer = 12,
        n_head = 12,
        d_head = 64,
        num_text_tokens = 10000,
        text_seq_len = 256,
        num_image_tokens = 222,
        image_seq_len = 576,
        image_size = 24,
        attn_pdrop = 0.1,
        resid_pdrop = 0.1,
        embd_pdrop = 0.1,
        ff_dropout = 0.1,
        attn_types = None,
        loss_img_weight = 7,
        loss_txt_weight = 7,
    ):
        super().__init__()

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        self.text_emb = nn.Embedding(num_text_tokens, n_embd)
        self.image_emb = nn.Embedding(num_image_tokens, n_embd)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, n_embd) # +1 for <bos> a.k.a <sos>
        # self.image_pos_emb = nn.Embedding(image_seq_len, n_embd)
        self.image_pos_emb = nn.Parameter(torch.zeros(1, image_seq_len, n_embd))
        # self.image_pos_emb = AxialPositionalEmbedding(n_embd, axial_shape=(image_size, image_size))

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_seq_len = seq_len
        self.total_tokens = total_tokens

        self.transformer = Transformer(
            dim = n_embd,
            causal = True,
            seq_len = seq_len,
            depth = n_layer,
            heads = n_head,
            dim_head = d_head,
            attn_dropout = attn_pdrop,
            resid_dropout = resid_pdrop,
            embd_dropout = embd_pdrop,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_size = image_size,
        )

        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, self.total_tokens),
        )

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, "n -> () n ()")
        logits_range = rearrange(logits_range, "d -> () () d")

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )

        self.register_buffer("logits_mask", logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight
        self.loss_txt_weight = loss_txt_weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        decoder,
        *,
        clip = None,
        filter_thres = 0.5,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None
    ):
        text_seq_len, image_seq_len, num_text_tokens = self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        batch = text.shape[0]
        text = text[:, :text_seq_len] # make sure text is within bounds
        out = text

        if exists(img):
            assert img.shape[1] == image_seq_len, f"input image must have the correct image size {image_seq_len}"

            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            assert num_img_tokens < image_seq_len, "number of initial image tokens for priming must be less than the total image token sequence length"

            trunc_img = img[:, :num_img_tokens]
            out = torch.cat((out, trunc_img), dim = -1)

        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self(text, image)[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)

            sample -= (num_text_tokens if is_image else 0) # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample), dim=-1)

        text_seq = out[:, :text_seq_len]
        img_seq = out[:, -image_seq_len:]

        scores = None
        if exists(clip):
            scores = clip(text_seq, img_seq, return_loss = False)

        img_seq = repeat(img_seq, "b p -> b p c", c=3)
        decoder = repeat(decoder, "p c -> b p c", b=batch)
        images = torch.gather(decoder, 1, img_seq)
        images = rearrange(images, "b (h w) c-> b c h w", h=24, w =24)
        images = images.float()

        return images, scores

    def forward(
        self,
        text,
        image = None,
        return_loss = False
    ):
        assert text.shape[-1] == self.text_seq_len, f"the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})"
        device, total_seq_len = text.device, self.total_seq_len

        text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        text = F.pad(text, (1, 0), value = 0) # add <bos>

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        seq_len = tokens.shape[1]
        
        image_len = image.shape[1]
        image_emb = self.image_emb(image)
        # image_emb += self.image_pos_emb(torch.arange(image_len, device = device))
        image_emb += self.image_pos_emb[:, :image_len, :]

        # image_emb += self.image_pos_emb(image_emb)

        tokens = torch.cat((tokens, image_emb), dim = 1)

        seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        out = self.transformer(tokens)
        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image

        logits_mask = self.logits_mask[:, :seq_len]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if not return_loss:
            return logits

        assert exists(image), "when training, image must be supplied"

        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim = 1)

        logits = rearrange(logits, "b n c -> b c n")

        loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
        loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])

        loss = (self.loss_txt_weight * loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + self.loss_txt_weight)
        return loss, loss_text, loss_img
