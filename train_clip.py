# CLIP and Tokenizer
from text2punks.text2punk import CLIP
from text2punks.loader import TextImageDataset
from text2punks.tokenizer import txt_tokenizer

import torch
import wandb  # Quit early if user doesn"t have wandb installed.
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader

import argparse
import time
from pathlib import Path


# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument("--clip_path", type=str,
                   help="path to your partially trained CLIP")

parser.add_argument("--image_text_path", type=str, required=True,
                    help="path to your path of images and text for learning the CLIP")

parser.add_argument("--truncate_captions", dest="truncate_captions", action="store_true",
                    help="Captions passed in which exceed the max token length will be truncated if this is set.")

parser.add_argument("--clip_output_file_name", type=str, default = "clip",
                    help="output_file_name")

parser.add_argument("--wandb_name", default="clip_train_transformer",
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

model_group.add_argument("--dim_latent", default = 512, type = int, help = "Model dimension")

model_group.add_argument("--dim_text", default = 512, type = int, help = "Text embedding dimension")

model_group.add_argument("--dim_image", default = 512, type = int, help = "Image embedding dimension")

model_group.add_argument("--text_enc_depth", default = 2, type = int, help = "Text encoder number of layers")

model_group.add_argument("--text_seq_len", default = 256, type = int, help = "Text sequence length")

model_group.add_argument("--text_heads", default = 8, type = int, help = "Text encoder number of heads")

model_group.add_argument("--num_visual_tokens", default = 222, type = int, help = "Number of visual tokens")

model_group.add_argument("--visual_enc_depth", default = 2, type = int, help = "Image encoder number of layers")

model_group.add_argument("--visual_heads", default = 8, type = int, help = "Image encoder number of heads")

model_group.add_argument("--visual_image_seq_len", default = 576, type = int, help = "The length of image sequence")

model_group.add_argument("--attn_pdrop", default = 0.1, type = float, help = "Model probability of dropout for attention layer")

model_group.add_argument("--resid_pdrop", default = 0.1, type = float, help = "Model probability of dropout for resnet layer")

model_group.add_argument("--embd_pdrop", default = 0.1, type = float, help = "Model probability of dropout for positional/word embedding layer")

model_group.add_argument("--ff_dropout", default = 0.1, type = float, help = "Feed forward dropout.")

model_group.add_argument("--attn_types", default = "full", type = str, help = "comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.")

args = parser.parse_args()


# quit early if you used the wrong path name

assert Path(args.image_text_path).exists(), f"The path {args.image_text_path} was not found."

# helpers fns

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


# constants

CLIP_PATH = args.clip_path
RESUME = exists(CLIP_PATH)
CLIP_OUTPUT_FILE_NAME = args.clip_output_file_name

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
MODEL_DIM = args.dim_latent
TEXT_DIM = args.dim_text
IMAGE_DIM = args.dim_image
TEXT_ENC_DEPTH = args.text_enc_depth
TEXT_SEQ_LEN = args.text_seq_len
TEXT_HEADS = args.text_heads
NUM_VISUAL_TOKENS = args.num_visual_tokens
VISUAL_ENC_DEPTH = args.visual_enc_depth
VISUAL_HEADS = args.visual_heads
VISUAL_IMAGE_SEQ_LEN = args.visual_image_seq_len
ATTN_PDROP = args.attn_pdrop
RESID_PDROP = args.resid_pdrop
EMBD_PDROP = args.embd_pdrop
FF_DROPOUT = args.ff_dropout

ATTN_TYPES = tuple(args.attn_types.split(","))

if RESUME:
    clip_path = Path(CLIP_PATH)
    assert clip_path.exists(), "CLIP model file does not exist"

    loaded_obj = torch.load(str(clip_path), map_location="cpu")

    clip_params, weights = loaded_obj["hparams"], loaded_obj["weights"]

    clip_params = dict(
        **clip_params
    )

else:
    clip_params = dict(
        dim_text = TEXT_DIM,
        dim_image = IMAGE_DIM,
        dim_latent = MODEL_DIM, 
        num_text_tokens = txt_tokenizer.vocab_size,
        text_enc_depth = TEXT_ENC_DEPTH,
        text_seq_len = TEXT_SEQ_LEN,
        text_heads = TEXT_HEADS,
        num_visual_tokens = NUM_VISUAL_TOKENS,
        visual_enc_depth = VISUAL_ENC_DEPTH,
        visual_heads = VISUAL_HEADS,
        visual_image_seq_len = VISUAL_IMAGE_SEQ_LEN,
        attn_pdrop = ATTN_PDROP,
        resid_pdrop = RESID_PDROP,
        embd_pdrop = EMBD_PDROP,
        ff_dropout = FF_DROPOUT,
        attn_types = ATTN_TYPES
    )


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

clip = CLIP(**clip_params)
if RESUME:
    clip.load_state_dict(weights)

clip = clip.cuda()


# optimizer

opt = Adam(get_trainable_params(clip), lr=LEARNING_RATE)

# experiment tracker

model_config = dict(
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    dim_text = TEXT_DIM,
    dim_image = IMAGE_DIM,
    dim_latent = MODEL_DIM,
    text_enc_depth = TEXT_ENC_DEPTH,
    text_heads = TEXT_HEADS,
    visual_enc_depth = VISUAL_ENC_DEPTH,
    visual_heads = VISUAL_HEADS,
    attn_pdrop = ATTN_PDROP,
    resid_pdrop = RESID_PDROP,
    embd_pdrop = EMBD_PDROP,
)

run = wandb.init(
    project=args.wandb_name,  # "clip_train_transformer" by default
    resume=RESUME,
    config=model_config,
)

def save_model(path):
    save_obj = {
        "hparams": clip_params,
    }

    save_obj = {
        **save_obj,
        "weights": clip.state_dict()
    }

    torch.save(save_obj, path)


# training
steps = 0
for epoch in range(EPOCHS):
    for i, (text, images) in enumerate(dl):
        if i % 10 == 0:
            t = time.time()

        text, images = map(lambda t: t.cuda(), (text, images))

        loss = clip(text, images, return_loss=True)
        loss = loss / ACCUM_STEPS
        loss.backward()

        if (i+1) % ACCUM_STEPS == 0:
            clip_grad_norm_(clip.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()

        # decay the learning rate based on our progress
        if LR_DECAY:
            if steps < WARMUP_ITER:
                # linear warmup
                lr_mult = FACTOR * (steps / WARMUP_ITER)
            else:
                # exponential learning rate decay
                lr_mult = FACTOR * max(0.057, (0.9992 ** (-WARMUP_ITER)) * (0.9992 ** steps))
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
                "lr": lr
            }

        if i % 10 == 9:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f"sample_per_sec - {sample_per_sec}")
        
        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(f"./{CLIP_OUTPUT_FILE_NAME}.pt")

        steps += 1
        wandb.log(log)

    # save trained model to wandb as an artifact every epoch"s end
    
    model_artifact = wandb.Artifact("trained-clip", type="model", metadata=dict(model_config))
    model_artifact.add_file(f"{CLIP_OUTPUT_FILE_NAME}.pt")
    run.log_artifact(model_artifact)

save_model(f"./{CLIP_OUTPUT_FILE_NAME}-final.pt")
wandb.save(f"./{CLIP_OUTPUT_FILE_NAME}-final.pt")
model_artifact = wandb.Artifact("trained-clip", type="model", metadata=dict(model_config))
model_artifact.add_file(f"{CLIP_OUTPUT_FILE_NAME}-final.pt")
run.log_artifact(model_artifact)

wandb.finish()
