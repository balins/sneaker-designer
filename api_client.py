import os
import sys
from string import Template
from time import sleep
import requests
from multiprocessing import Pool

from urllib3.exceptions import NewConnectionError

API_ENDPOINT = Template("https://api.thesneakerdatabase.com/v1/sneakers?limit=100&page=$page")


def download_images(output_path: str, img_size="s", limit=float("inf"), starting_page=0):
    size_arg = get_img_size_arg(img_size)
    os.makedirs(output_path, exist_ok=True)
    page = starting_page
    n_images = 0
    p = Pool()

    while n_images < limit and (urls := get_image_urls(page, size_arg)):
        if n_images + len(urls) >= limit:
            urls = urls[:limit - n_images]

        p.starmap_async(save_image, [(url, output_path) for url in urls])
        n_images += len(urls)
        page += 1

    p.close()
    p.join()


def get_image_urls(page: int, img_size: str) -> list:
    sneakers = requests.get(API_ENDPOINT.substitute(page=page)).json()["results"]
    return [img_url for sneaker in sneakers if (img_url := sneaker["media"][img_size])]


def save_image(img_url: str, output_path: str):
    def hold_on(time_in_s):
        print(f"Next attempt in {time_in_s}s...")
        sleep(time_in_s)

    for attempts in (1, 2, 3):
        try:
            image = requests.get(img_url)
            break
        except NewConnectionError:
            print(f"[{attempts}/3] Failed to fetch image from {img_url} due to connection error.", file=sys.stderr)
            if attempts < 3:
                hold_on(5*attempts)
    else:
        print(f"All trials failed. Returning...")
        return

    filename = filename_from_url(img_url)
    filepath = os.path.join(output_path, filename)

    with open(filepath, 'wb') as f:
        print(f"Saving {filename}...")
        f.write(image.content)


def get_img_size_arg(img_size: str) -> str:
    valid_sizes = {
        ("small", "s"): "thumbUrl",
        ("medium", "m"): "imageUrl",
        ("large", "l"): "smallImageUrl"
    }

    for size_key, size_arg in valid_sizes.items():
        if img_size in size_key:
            return size_arg
    else:
        raise ValueError(f"Argument size has to be in {list(valid_sizes.keys())}. Got '{img_size}' instead.")


def filename_from_url(img_url: str) -> str:
    filename = img_url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]

    return filename


if __name__ == "__main__":
    # todo cmd args?
    print("Downloading all available images...")
    download_images("images")
