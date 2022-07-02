from utils import model_loader, generate_image

import tweepy

import time
import schedule
import numpy as np
from PIL import Image


# Functions

def create_api():
    twitter_auth_keys = { 
        "consumer_key"        : "YOUR_CONSUMER_KEY",
        "consumer_secret"     : "YOUR_CONSUMER_SECRET",
        "access_token"        : "YOUR_ACCESS_TOKEN",
        "access_token_secret" : "ACCESS_TOKEN_SECRET"
    }

    try:
        # Authenticate to Twitter
        auth = tweepy.OAuthHandler(twitter_auth_keys["consumer_key"], twitter_auth_keys["consumer_secret"])
        auth.set_access_token(twitter_auth_keys["access_token"], twitter_auth_keys["access_token_secret"])
        # Create API object
        api = tweepy.API(auth, wait_on_rate_limit=True)
        return api

    except:
        print("Problems with the KEYS")
        return None


def retrieve_last_seen_id(file_name="./mention_id.txt"):
    with open(file_name, "r") as f:
        last_seen_id = f.read().strip()
    
    return last_seen_id


def store_last_seen_id(id, file_name="./mention_id.txt"):
    with open(file_name, "w") as f:
        f.write(str(id))


def extract_text(s):
    return " ".join([part for part in s.split() if not (part.startswith("#") or part.startswith("@"))])


def resize_image(image_path, size=600):
    img = Image.open(image_path)
    resized_img = img.resize((size, size), resample=Image.NEAREST)
    resized_img.save(image_path)


# Hair
Hair = np.asarray(["Hoodie", "Mohawk Thin", "Mohawk", "Mohawk Dark", "Crazy Hair", "Red Mohawk", "Stringy Hair", "Messy Hair", "Wild Hair", "Frumpy Hair", "Peak Spike", "Purple Hair", "Dark Hair", "Straight Hair", "Straight Hair Dark", "Clown Hair Green", "Vampire Hair", "Blonde Bob", "Half Shaved", "Straight Hair Blonde", "Wild Blonde", "Wild White Hair", "Blonde Short", "Pigtails", "Orange Side"])
# Beard
Beard = np.asarray(["Muttonchops", "Shaved Head", "Goat", "Normal Beard", "Normal Beard Black", "Mustache", "Luxurious Beard", "Chinstrap", "Front Beard", "Handlebars", "Front Beard Dark", "Big Beard", "Shadow Beard"])
# Cap
Cap = np.asarray(["Bandana", "Knitted Cap", "Headband", "Cap", "Do-rag", "Cap Forward", "Police Cap", "Fedora", "Tassle Hat", "Cowboy Hat", "Top Hat", "Pink With Hat", "Tiara", "Pilot Helmet", "Beanie"])
# Glasses
Glasses = np.asarray(["Nerd Glasses", "Big Shades", "Horned Rim Glasses", "Regular Shades", "Classic Shades", "Small Shades", "VR", "Eye Mask", "3D Glasses", "Eye Patch"])
# Lipstick
Lipstick = np.asarray(["Hot Lipstick", "Purple Lipstick", "Black Lipstick"])
# Clown and shadow Eyes
Shadow_Eyes = np.asarray(["Green Eye Shadow", "Blue Eye Shadow", "Purple Eye Shadow", "Clown Eyes Blue", "Clown Eyes Green"])
# Chain
Chain = np.asarray(["Gold Chain", "Silver Chain", "Choker"])
# Spots
Spots = np.asarray(["Rosy Cheeks", "Spots", "Mole"])
# Cigarette
Cigarette = np.asarray(["Cigarette", "Pipe", "Vape"])
# Welding Goggles
Welding_Goggles = np.asarray(["Welding Goggles"])
# Frown_Smile
Frown_Smile = np.asarray(["Frown", "Smile"])
# Medical Mask
Medical_Mask = np.asarray(["Medical Mask"])
# Clown Nose
Clown_Nose = np.asarray(["Clown Nose"])
# Buck Teeth
Buck_Teeth = np.asarray(["Buck Teeth"])
# Earring
Earring = np.asarray(["Earring"])

num_of_attributes = np.asarray(["0 Attributes", "1 Attributes", "2 Attributes", "3 Attributes", "4 Attributes", "5 Attributes", "6 Attributes", "7 Attributes"])
categories = np.asarray(["Alien", "Ape", "Zombie", "Male", "Female"])

attributes = [Hair, Beard, Cap, Glasses, Lipstick, Shadow_Eyes, Chain, Spots, Cigarette, Welding_Goggles, Frown_Smile, Medical_Mask, Clown_Nose, Buck_Teeth, Earring]
    

def template_caption(category, attributes=None):

    ana = "A" if category not in ["Ape", "Alien"] else "An"
    if attributes is not None:
        Text_caption = [
            f"{ana} {category} cryptopunk that has {len(attributes)} attributes, ",
            f"Funny looking {category} cryptopunk with {len(attributes)} attributes, ", 
            f"A low resolution photo of punky-looking {category} that has {len(attributes)} attributes, "               
        ]
    else:
        Text_caption = [
            f"{ana} {category} cryptopunk that has with 0 attributes."
            f"Simple looking {category} cryptopunk made of 0 attributes." 
            f"A low resolution photo of punky-looking {category} that has 0 attributes."              
        ]

        caption = Text_caption[np.random.choice(range(len(Text_caption)), size=1, replace=False, p=None)[0]]
        return caption

    caption = Text_caption[np.random.choice(range(len(Text_caption)), size=1, replace=False, p=None)[0]]

    for a in attributes[:-1]:
        caption += "a " + a + ", "
    
    caption += "and a " + attributes[-1]
    
    return caption


def generate_caption():
    distro = [0.0008, 0.0333, 0.356, 0.4501, 0.1420, 0.0166, 0.0011, 0.0001]
    att_count = np.random.choice(num_of_attributes, size=None, replace=False, p=distro)
    category = np.random.choice(categories, size=None, replace=False, p=None)

    if att_count == "0 Attributes":
        # zero attributes
        idx = None
    elif att_count == "1 Attributes":
        # one attribute
        idx = np.random.choice(range(15), size=1, replace=False, p=None)
    elif att_count == "2 Attributes":
        # two attributes
        idx = np.random.choice(range(15), size=2, replace=False, p=None)
    elif att_count == "3 Attributes":
        # three attributes
        idx = np.random.choice(range(15), size=3, replace=False, p=None)
    elif att_count == "4 Attributes":
        # four attributes
        idx = np.random.choice(range(15), size=4, replace=False, p=None)
    elif att_count == "5 Attributes":
        # five attributes
        idx = np.random.choice(range(15), size=5, replace=False, p=None)
    elif att_count == "6 Attributes":
        # six attributes
        idx = np.random.choice(range(15), size=6, replace=False, p=None)
    else:
        # seven attributes
        idx = np.random.choice(range(15), size=7, replace=False, p=None)


    if idx is not None:
        attr = [attributes[i] for i in idx]
        features = [np.random.choice(feat, size=None, replace=False, p=None) for feat in attr]
        caption = template_caption(category, features)
    else:
        caption = template_caption(category, attributes=None)

    return caption


attributes_list = [feature for att in attributes for feature in att]
def check_prompt(prompt):
    return any(att in prompt for att in attributes_list)


def tweet_image_reply(api):

    FILE_NAME = "./mention_id.txt"
    last_seen_id = retrieve_last_seen_id(FILE_NAME)
    mentions = api.mentions_timeline(last_seen_id, tweet_mode="extended")

    for mention in reversed(mentions):
        last_seen_id = mention.id
        store_last_seen_id(last_seen_id, FILE_NAME)
        tweet = extract_text(mention.full_text)

        if check_prompt(tweet):
            print("Generating Image from Text")
            caption = tweet
            generate_image(caption, last_seen_id, text2punk, clip)

            # Upload images and get media_ids
            filename_img = f"./outputs/{last_seen_id}/0.png"
            filename_sim = f"./sims/{last_seen_id}_.png"
            resize_image(filename_img)
            media_img = api.media_upload(filename_img)
            media_sim = api.media_upload(filename_sim)

            # Post tweet with image
            try:
                api.create_favorite(mention.id)
                api.update_status(status=f"@{mention.user.screen_name} {caption} #Text2Cryptopunks #NFTcommuity #cryptopunks #aiart #MachineLearning", in_reply_to_status_id=mention.id, media_ids=[media_img.media_id, media_sim.media_id])
                print("Just tweeted")

            except:
                print("Tweets rate limit reached")

        else:
            print("Generating Image from Text")
            caption = generate_caption()
            generate_image(caption, caption.replace(" ", "_")[:(100)], text2punk, clip)

            # Upload images and get media_ids
            filename_img = f"./outputs/{caption.replace(" ", "_")[:(100)]}/0.png"
            resize_image(filename_img)
            filename_sim = f"./sims/{caption.replace(" ", "_")[:(100)]}_.png"
            media_img = api.media_upload(filename_img)
            media_sim = api.media_upload(filename_sim)

            # Post tweet with image
            try:
                api.update_status(status=f"tweet somthing like this "{caption}"", media_ids=[media_img.media_id, media_sim.media_id])
                print("Just tweeted an image")

            except:
                print("Tweets rate limit reached")            


def tweet_image(api):

    print("Generating Image from Text")
    caption = generate_caption()
    generate_image(caption, caption.replace(" ", "_")[:(100)], text2punk, clip)

    # Upload images and get media_ids
    filename_img = f"./outputs/{caption.replace(" ", "_")[:(100)]}/0.png"
    resize_image(filename_img)
    filename_sim = f"./sims/{caption.replace(" ", "_")[:(100)]}_.png"
    media_img = api.media_upload(filename_img)
    media_sim = api.media_upload(filename_sim)

    # Post tweet with image
    try:
        api.update_status(status=f"{caption} #Text2Cryptopunks #NFTs #NFTcommuity #cryptopunks #aiart #generativeart #MachineLearning", media_ids=[media_img.media_id, media_sim.media_id])
        print("Just tweeted an image")

    except:
        print("Tweets rate limit reached")


# retrive all mentions id 
def retrieve_mentionsId(api):
    mentions = api.mentions_timeline()

    for mention in mentions:
        print(mention.id)


if __name__ == "__main__":

    api = create_api()

    # Tweet every 60 minutes
    schedule.every(60).minutes.do(tweet_image, api)

    t2p_path, clip_path = "./Text2Punk-final-7.pt", "clip-final.pt"
    text2punk, clip = model_loader(t2p_path, clip_path)

    while True:

        schedule.run_pending()
        time.sleep(3)

        tweet_image_reply(api)
