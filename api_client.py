import os
import sys
from multiprocessing import Pool
from string import Template
from time import sleep
from urllib.parse import urlparse
import requests
from urllib3.exceptions import NewConnectionError

API_ENDPOINT = Template("https://api.thesneakerdatabase.com/v1/sneakers?limit=100&page=$page")
inf = sys.maxsize


def download_images(output_dir: str, img_size="s", limit=inf, starting_page=0):
    size_arg = get_img_size_arg(img_size)
    if limit < 0 or starting_page < 0:
        raise ValueError("Starting page number and limit must not be negative!")

    os.makedirs(output_dir, exist_ok=True)

    pool = Pool()
    image_urls = setup_image_urls_generator(size_arg, starting_page)
    n_images = 0

    while n_images < limit:
        try:
            urls = next(image_urls)
        except StopIteration:
            break

        urls = urls[:min(len(urls), limit - n_images)]
        pool.starmap_async(save_image, [(url, output_dir) for url in urls])
        n_images += len(urls)

    pool.close()
    pool.join()

    print(f"Successfully downloaded {n_images} images.", file=sys.stderr)


def save_image(img_url: str, output_dir: str):
    for attempt in (1, 2, 3):
        try:
            image = requests.get(img_url)
            break
        except NewConnectionError:
            print(f"[{attempt}/3] Failed to fetch image from {img_url} due to connection error.", file=sys.stderr)
            if attempt < 3:
                print(f"Next attempt in {5 * attempt}s...")
                sleep(5 * attempt)
    else:
        print("All trials failed. Returning...")
        return

    filename = os.path.basename(urlparse(img_url).path)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'wb') as f:
        print(f"Saving {filename}...")
        f.write(image.content)


def setup_image_urls_generator(size_arg: str, starting_page: int) -> iter:
    def image_urls_generator() -> list:
        page = starting_page
        while True:
            sneakers = requests.get(API_ENDPOINT.substitute(page=page)).json()["results"]
            if not sneakers:
                print(f"API ran out of sneakers on page {page}.", file=sys.stderr)
                break

            image_urls = [img_url for sneaker in sneakers if (img_url := sneaker["media"][size_arg])]
            page += 1

            yield image_urls

    return image_urls_generator()


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


if __name__ == "__main__":
    # todo cmd args?
    print("Downloading all available images...")
    download_images("images", starting_page=297)
