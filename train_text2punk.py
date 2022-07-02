# Text2Punks and Tokenizer
from text2punks.text2punk import Text2Punks
from text2punks.loader import TextImageDataset
from text2punks.tokenizer import txt_tokenizer

import torch
import wandb  # Quit early if user doesn"t have wandb installed.
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from PIL import Image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import argparse
import time
from pathlib import Path


# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument("--text2punk_path", type=str,
                   help="path to your partially trained Text2Punk")

parser.add_argument("--image_text_path", type=str, required=True,
                    help="path to your path of images and text for learning the Text2Punk")

parser.add_argument("--truncate_captions", default=True, type=bool,
                    help="Captions passed in which exceed the max token length will be truncated if this is set.")

parser.add_argument("--text2punks_output_file_name", type=str, default = "Text2Punk",
                    help="output_file_name")

parser.add_argument("--wandb_name", default="text2punk_train_transformer",
                    help="Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`")


train_group = parser.add_argument_group("Training settings")

train_group.add_argument("--epochs", default = 20, type = int, help = "Number of epochs")

train_group.add_argument("--batch_size", default = 4, type = int, help = "Batch size")

train_group.add_argument("--learning_rate", default = 3e-4, type = float, help = "Learning rate")

train_group.add_argument("--clip_grad_norm", default = 0.5, type = float, help = "Clip gradient norm")

train_group.add_argument("--lr_decay", default = False, type = bool, help = "learning rate decay params: linear warmup followed by cosine decay to 10% of original")

train_group.add_argument("--factor", default = 1, type = float, help = "learning rate decay multiplier factor")

train_group.add_argument("--warm_up_iter", default = 2000, type = int, help = "Warm-up iterations in the beginning of training")

train_group.add_argument("--accumulation_steps", default = 4, type = int, help = "number of steps to accumulate gration")

train_group.add_argument("--save_every_n_steps", default = 500, type = int, help = "Save a checkpoint every n steps")

train_group.add_argument("--codebook", type=str, help="path to image encoding/decodeing codebook")


model_group = parser.add_argument_group("Model settings")

model_group.add_argument("--n_embd", default = 512, type = int, help = "Model dimension")

model_group.add_argument("--n_layer", default = 2, type = int, help = "Model number of layers")

model_group.add_argument("--n_head", default = 8, type = int, help = "Model number of heads")

model_group.add_argument("--d_head", default = 64, type = int, help = "Model dimension of head")

model_group.add_argument("--text_seq_len", default = 256, type = int, help = "Text sequence length")

model_group.add_argument("--image_seq_len", default = 576, type = int, help = "The length of image sequence")

model_group.add_argument("--num_image_tokens", default = 222, type = int, help = "Number of image tokens")

model_group.add_argument("--image_size", default = 24, type = int, help = "Image size")

model_group.add_argument("--attn_pdrop", default = 0.1, type = float, help = "Model probability of dropout for attention layer")

model_group.add_argument("--resid_pdrop", default = 0.1, type = float, help = "Model probability of dropout for resnet layer")

model_group.add_argument("--embd_pdrop", default = 0.1, type = float, help = "Model probability of dropout for positional/word embedding layer")

model_group.add_argument("--ff_dropout", default = 0.1, type = float, help = "Feed forward dropout.")

model_group.add_argument("--attn_types", default = "full", type = str, help = "comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.")

model_group.add_argument("--loss_img_weight", default = 7, type = int, help = "Image loss weight")

model_group.add_argument("--loss_txt_weight", default = 7, type = int, help = "Text loss weight")


args = parser.parse_args()

# quit early if you used the wrong path name

assert Path(args.image_text_path).exists(), f"The path {args.image_text_path} was not found."

# helpers fns

def exists(val):
    return val is not None

# constants

TEXT2PUNK_PATH = args.text2punk_path
RESUME = exists(TEXT2PUNK_PATH)
TEXT2PUNKS_OUTPUT_FILE_NAME = args.text2punks_output_file_name

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
LR_DECAY = args.lr_decay
FACTOR = args.factor
WARMUP_ITER = args.warm_up_iter
ACCUM_STEPS = args.accumulation_steps
SAVE_EVERY_N_STEPS = args.save_every_n_steps

CODEBOOK = args.codebook
MODEL_DIM = args.n_embd
LAYERS = args.n_layer
HEADS = args.n_head
HEAD_DIM = args.d_head
TEXT_SEQ_LEN = args.text_seq_len
IMAGE_SEQ_LEN = args.image_seq_len
NUM_IMAGE_TOKENS = args.num_image_tokens
IMAGE_SIZE = args.image_size
ATTN_PDROP = args.attn_pdrop
RESID_PDROP = args.resid_pdrop
EMBD_PDROP = args.embd_pdrop
FF_DROPOUT = args.ff_dropout
LOSS_IMG_WEIGHT = args.loss_img_weight
LOSS_TXT_WEIGHT = args.loss_txt_weight

ATTN_TYPES = tuple(args.attn_types.split(","))

if RESUME:
    text2punk_path = Path(TEXT2PUNK_PATH)
    assert text2punk_path.exists(), "Text2Punks model file does not exist"

    loaded_obj = torch.load(str(text2punk_path), map_location="cpu")

    text2punk_params, weights = loaded_obj["hparams"], loaded_obj["weights"]
    opt_state = loaded_obj.get("opt_state")

    text2punk_params = dict(
        **text2punk_params
    )

    resume_epoch = loaded_obj.get("epoch", 0)

else:
    text2punk_params = dict(
        n_embd=MODEL_DIM,
        n_layer=LAYERS,
        n_head=HEADS,
        d_head=HEAD_DIM,
        num_text_tokens=txt_tokenizer.vocab_size,
        text_seq_len=TEXT_SEQ_LEN,
        num_image_tokens=NUM_IMAGE_TOKENS,
        image_seq_len=IMAGE_SEQ_LEN,
        image_size=IMAGE_SIZE,
        attn_pdrop=ATTN_PDROP,
        resid_pdrop=RESID_PDROP,
        embd_pdrop=EMBD_PDROP,
        ff_dropout=FF_DROPOUT,
        attn_types=ATTN_TYPES,
        loss_img_weight=LOSS_IMG_WEIGHT,
        loss_txt_weight=LOSS_TXT_WEIGHT,
    )

    resume_epoch = 0


# helper fns

def configure_optimizers(model, weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95)):
    # separate out all parameters to those that will and won"t experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn # full param name

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter
    no_decay.add("image_pos_emb")


    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


# create image encoder, to be modified later

codebook = torch.load(CODEBOOK)
rgb_color_str = [str(rgb) for rgb in codebook.numpy()]
code = [i for i in range(0, 222)]
img_tokenizer = dict(zip(rgb_color_str, code))

# create dataset and dataloader

ds = TextImageDataset(
    args.image_text_path,
    text_len=TEXT_SEQ_LEN,
    truncate_captions=args.truncate_captions,
    text_tokenizer=txt_tokenizer,
    image_tokenizer=img_tokenizer,
    shuffle=True,
)

assert len(ds) > 0, "dataset is empty"
print(f"{len(ds)} image-text pairs found for training")

dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

text2punk = Text2Punks(**text2punk_params)
if RESUME:
    text2punk.load_state_dict(weights)

text2punk = text2punk.cuda()

# optimizer

opt = configure_optimizers(text2punk, weight_decay=0.045, learning_rate=LEARNING_RATE, betas=(0.9, 0.96))

if RESUME:
    opt.load_state_dict(opt_state)


# experiment tracker

model_config = dict(
    n_embd = MODEL_DIM,
    n_layer = LAYERS,
    n_head = HEADS,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    attn_pdrop = ATTN_PDROP,
    resid_pdrop = RESID_PDROP,
    embd_pdrop = EMBD_PDROP,
)

run = wandb.init(
    project=args.wandb_name,  # "text2punk_train_transformer" by default
    resume=RESUME,
    config=model_config,
)

def save_model(path, epoch=0):
    save_obj = {
        "hparams": text2punk_params,
        "epoch": epoch
    }

    save_obj = {
        **save_obj,
        "weights": text2punk.state_dict(),
        "opt_state": opt.state_dict()
    }

    torch.save(save_obj, path)


# Saves a checkpoint before training begins to fail early when mis-configured. 

save_model(f"./{TEXT2PUNKS_OUTPUT_FILE_NAME}.pt", epoch=resume_epoch)

# training

steps = 0
for epoch in range(resume_epoch, EPOCHS):
    for i, (texts, images) in enumerate(dl):
        if i % 10 == 0:
            t = time.time()

        texts, images = map(lambda t: t.cuda(), (texts, images))

        loss, text_loss, image_loss = text2punk(texts, images, return_loss=True)
        loss = loss / ACCUM_STEPS
        loss.backward()

        if (i+1) % ACCUM_STEPS == 0:
            clip_grad_norm_(text2punk.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()

        # decay the learning rate based on our progress

        if LR_DECAY:
            if steps < WARMUP_ITER:
                # linear warmup
                lr_mult = FACTOR * (steps / WARMUP_ITER)
            else:
                # exponential learning rate decay
                lr_mult = FACTOR * max(0.057, (0.999 ** (-WARMUP_ITER)) * (0.999 ** steps))
            lr = LEARNING_RATE * lr_mult
            for param_group in opt.param_groups:
                param_group["lr"] = lr
        else:
            lr = LEARNING_RATE
            

        log = {}

        if i % 10 == 0:
            print(f"epoch - {epoch},", f"step - {i},", f"loss - {loss.item()}")

            log = {
                **log,
                "epoch": epoch,
                "iter": i,
                "loss": loss.item(),
                "text_loss": text_loss.item(),
                "image_loss": image_loss.item(),
                "lr": lr
            }

        if i % 100 == 0:
            sample_text = texts[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = txt_tokenizer.decode(token_list)

            image, _ = text2punk.generate_images(sample_text, codebook.cuda(), clip = None, filter_thres = 0.7, temperature = 1.25)

            log = {
                **log,
            }

            grid = make_grid(image, value_range=(0, 255), normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            pil_image = Image.fromarray(ndarr)

            log["image"] = wandb.Image(pil_image, caption=decoded_text)

        if i % 10 == 9:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f"sample_per_sec - {sample_per_sec}")

        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(f"./{TEXT2PUNKS_OUTPUT_FILE_NAME}.pt", epoch=epoch)

        steps += 1
        wandb.log(log)

    # save trained model to wandb as an artifact every epoch"s end
    model_artifact = wandb.Artifact("trained-text2punk", type="model", metadata=dict(model_config))
    model_artifact.add_file(f"{TEXT2PUNKS_OUTPUT_FILE_NAME}.pt")
    run.log_artifact(model_artifact)


save_model(f"./{TEXT2PUNKS_OUTPUT_FILE_NAME}-final.pt", epoch=epoch)
wandb.save(f"./{TEXT2PUNKS_OUTPUT_FILE_NAME}.pt")
model_artifact = wandb.Artifact("trained-text2punk", type="model", metadata=dict(model_config))
model_artifact.add_file(f"{TEXT2PUNKS_OUTPUT_FILE_NAME}-final.pt")
run.log_artifact(model_artifact)

wandb.finish()