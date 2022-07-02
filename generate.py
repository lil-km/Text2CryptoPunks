# Text2Punks and Tokenizer
from text2punks.text2punk import Text2Punks, CLIP
from text2punks.tokenizer import txt_tokenizer

import torch
from einops import repeat

from PIL import Image
from torchvision.utils import save_image

from tqdm import tqdm
import argparse
from pathlib import Path


# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument("--text2punk_path", type = str, required = True,
                    help="path to your trained Text2Punks")

parser.add_argument("--clip_path", type = str, required = False,
                    help="path to your trained CLIP")

parser.add_argument("--text", type = str, required = True,
                    help="your text prompt")

parser.add_argument("--codebook", type=str,
                    help="path to image encoding/decodeing codebook")

parser.add_argument("--num_images", type = int, default = 5, required = False,
                    help="number of images")

parser.add_argument("--batch_size", type = int, default = 4, required = False,
                    help="batch size")

parser.add_argument("--top_k", type = float, default = 0.9, required = False,
                    help="top k filter threshold")

parser.add_argument("--temperature", type = float, default = 0.9, required = False,
                    help="higher temperatures work better (e.g. 0.7 - 1.3)")

parser.add_argument("--outputs_dir", type = str, default = "./outputs", required = False,
                    help="output directory")


args = parser.parse_args()


# helper fns

def exists(val):
    return val is not None


# load TEXT2PUNKS

text2punk = None
if exists(args.text2punk_path):
    text2punk_path = Path(args.text2punk_path)
    assert text2punk_path.exists(), "trained Text2Punks must exist"

    load_obj = torch.load(str(text2punk_path))
    text2punks_params, weights = load_obj.pop("hparams"), load_obj.pop("weights")

    text2punk = Text2Punks(**text2punks_params).cuda()
    text2punk.load_state_dict(weights)


# load pre-trained clip model 

clip = None
if exists(args.clip_path):
    clip_path = Path(args.clip_path)
    assert clip_path.exists(), "trained CLIP must exist"

    load_obj = torch.load(str(clip_path))
    clip_params, weights = load_obj.pop("hparams"), load_obj.pop("weights")

    clip = CLIP(**clip_params).cuda()
    clip.load_state_dict(weights)


# load decoder

codebook = torch.load(args.codebook)

# generate images

texts = args.text.split("|")

for text in tqdm(texts):
    text = txt_tokenizer.tokenize(text, text2punk.text_seq_len).cuda()

    text = repeat(text, "() n -> b n", b = args.num_images)

    img_outputs = []
    score_outputs = []

    if exists(clip):
        for text_chunk in tqdm(text.split(args.batch_size), desc = f"generating images for - {text}"):
            images, scores = text2punk.generate_images(text_chunk, codebook.cuda(), clip = clip, filter_thres = args.top_k, temperature = args.temperature)
            img_outputs.append(images)
            score_outputs.append(scores)

        img_outputs = torch.cat(img_outputs)
        score_outputs = torch.cat(score_outputs)

        similarity = score_outputs.softmax(dim=-1)
        values, indices = similarity.topk(args.num_images)

        img_outputs = img_outputs[indices]
        score_outputs = score_outputs[indices]

    else:
        for text_chunk in tqdm(text.split(args.batch_size), desc = f"generating images for - {text}"):
            images, _ = text2punk.generate_images(text_chunk, codebook.cuda(), clip = None, filter_thres = args.top_k, temperature = args.temperature)
            img_outputs.append(images)

        img_outputs = torch.cat(img_outputs)

    # save all images

    outputs_dir = Path(args.outputs_dir) / args.text.replace(" ", "_")[:(100)]
    outputs_dir.mkdir(parents = True, exist_ok = True)

    for i, (image, score) in tqdm(enumerate(zip(img_outputs, score_outputs)), desc = "saving images"):
        save_image(image, outputs_dir / f"{i}.png", value_range=(0, 255), normalize=True)
        print()
        print(f"Image {i} score is: {score}")

    print(f"created {args.num_images} images at "{str(outputs_dir)}"")
