# Text2Punks and Tokenizer
from text2punks.text2punk import Text2Punks, CLIP
from text2punks.tokenizer import txt_tokenizer

import torch
from einops import repeat

from PIL import Image
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load decoder
codebook = torch.load("./text2punks/data/codebook.pt")

outputs_folder = "./outputs"
sim_folder = "./sims"
batch_size = 32
num_images = 32

# create similarity directory 
sim_dir = Path(sim_folder)
sim_dir.mkdir(exist_ok = True)

# nobs to tune
top_k = 0.8
temperature = 1.25

# helper fns

def exists(val):
    return val is not None


def model_loader(text2punk_path, clip_path):
    # load pre-trained TEXT2PUNKS model

    text2punk_path = Path(text2punk_path)
    assert text2punk_path.exists(), "trained Text2Punks must exist"

    load_obj = torch.load(str(text2punk_path), map_location=torch.device(device))
    text2punks_params, weights = load_obj.pop("hparams"), load_obj.pop("weights")

    text2punk = Text2Punks(**text2punks_params).to(device)
    text2punk.load_state_dict(weights)

    # load pre-trained CLIP model

    clip_path = Path(clip_path)
    assert clip_path.exists(), "trained CLIP must exist"

    load_obj = torch.load(str(clip_path), map_location=torch.device(device))
    clip_params, weights = load_obj.pop("hparams"), load_obj.pop("weights")

    clip = CLIP(**clip_params).to(device)
    clip.load_state_dict(weights)

    return text2punk, clip


def generate_image(tweet, reply_id, text2punk_model, clip_model):
    text = txt_tokenizer.tokenize(tweet, text2punk_model.text_seq_len, truncate_text=True).to(device)

    text = repeat(text, "() n -> b n", b = num_images)

    img_outputs = []
    score_outputs = []

    for text_chunk in text.split(batch_size):
        images, scores = text2punk_model.generate_images(text_chunk, codebook.to(device), clip = clip_model, filter_thres = top_k, temperature = temperature)
        img_outputs.append(images)
        score_outputs.append(scores)

    img_outputs = torch.cat(img_outputs)
    score_outputs = torch.cat(score_outputs)

    similarity = score_outputs.softmax(dim=-1)
    values, indices = similarity.topk(num_images)

    img_outputs = img_outputs[indices]
    score_outputs = score_outputs[indices]

    # save all images
    outputs_dir = Path(outputs_folder) / str(reply_id)
    outputs_dir.mkdir(parents = True, exist_ok = True)


    for i, (image, score) in enumerate(zip(img_outputs, score_outputs)):
        save_image(image, outputs_dir / f"{i}.png", value_range=(0, 255), normalize=True)

    print(f"Tweet id {reply_id} created {num_images} images at "{str(outputs_dir)}"\n")


    fig, ax = plt.subplots(4, 8, sharex="col", sharey="row", figsize=(20, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"{tweet}", size=15)

    for i in range(4):
        for j in range(8):
            img = Image.open(outputs_dir / f"{i*8 + j}.png").convert("RGB")
            img = np.array(img)
            ax[i, j].imshow(img)
            ax[i, j].set_title(f"{score_outputs[i*8 + j]:.4f}", fontsize=15)
            ax[i, j].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(f"{str(sim_dir)}/{reply_id}_.png", dpi=300)
    
    plt.close()