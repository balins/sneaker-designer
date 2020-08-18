import os
import sys

import requests
from string import Template
from multiprocessing import Process

from urllib3.exceptions import NewConnectionError


def get_size_arg(size: str):
    if size in ("small", "s"):
        return "thumbUrl"
    elif size in ("medium", "m"):
        return "smallImageUrl"
    elif size in ("large", "l"):
        return "imageUrl"
    else:
        raise ValueError(f"Argument size has to be one of 'small' ('s'), 'medium' ('m'), 'large' ('l'). Got '{size}'")


def download_images(output_path, size='s', limit=float("inf"), skip_existing=True):
    size_arg = get_size_arg(size)
    os.makedirs(output_path, exist_ok=True)
    api_endpoint = Template("https://api.thesneakerdatabase.com/v1/sneakers?limit=100&page=$page")
    page = n_images = 0

    while data := requests.get(api_endpoint.substitute(page=page)).json()["results"]:
        for entrance in data:
            if img_url := entrance["media"][size_arg]:
                p = Process(target=save_image, args=(img_url, output_path, skip_existing))
                p.start()
                n_images += 1
                if n_images == limit:
                    try:
                        p.join()
                    except AssertionError:
                        pass
                    return
        page += 1


def save_image(img_url, output_path, skip_existing=True):
    filename = img_url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]

    filepath = os.path.join(output_path, filename)
    if skip_existing and os.path.exists(filepath):
        print(f"File {filename} already exists. Skipping...")
        return

    try:
        image = requests.get(img_url)
    except NewConnectionError:
        print(f"Failed to fetch image from {img_url} due to connection error.", file=sys.stderr)
        return

    with open(filepath, 'wb') as f:
        print(f"Saving {filename}...")
        f.write(image.content)


if __name__ == "__main__":
    # todo thread pool? cmd args?
    print("Downloading all available images...")
    download_images("images")
