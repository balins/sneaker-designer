import asyncio
import itertools
import os
import sys
from string import Template

import aiofiles
import aiohttp

API_ENDPOINT = Template("https://api.thesneakerdatabase.com/v1/sneakers?limit=100&page=$page")

img_size_args = {
    **dict.fromkeys(("s", "small"), "thumbUrl"),
    **dict.fromkeys(("medium", "m"), "imageUrl"),
    **dict.fromkeys(("large", "l"), "smallImageUrl")
}


async def start_download(output_dir="images", starting_page=0, limit=sys.maxsize, img_size="s"):
    try:
        size_arg = img_size_args[img_size]
    except IndexError:
        raise ValueError(f"Argument size has to be in {list(img_size_args.keys())}. Got '{img_size}' instead.")
    if starting_page < 0:
        raise ValueError("Starting page number must not be negative!")

    limit = max(10, limit)

    os.makedirs(output_dir, exist_ok=True)

    await download_images(output_dir, starting_page, limit, size_arg)


async def download_images(output_dir="images", starting_page=0, limit=sys.maxsize, size_arg=img_size_args["s"]):
    download_queue = asyncio.Queue()
    save_queue = asyncio.Queue()
    session = aiohttp.ClientSession()

    tasks = [
        asyncio.Task(sneaker_jsons(size_arg, session, starting_page, download_queue)),
        asyncio.Task(download_image(limit, session, download_queue, save_queue)),
        asyncio.Task(save_image(output_dir, save_queue))
    ]

    await asyncio.gather(*tasks)

    await session.close()


async def sneaker_jsons(size_arg: str, session: aiohttp.ClientSession, starting_page: int,
                        download_queue: asyncio.Queue):
    for page in itertools.count(starting_page):
        async with session.get(API_ENDPOINT.substitute(page=page)) as resp:
            content = await resp.json()
            if len(content["results"]) > 0:
                image_urls = [img_url["media"][size_arg] for img_url in content["results"]]
                await download_queue.put(image_urls)
            else:
                download_queue.put_nowait(None)
                break


async def download_image(limit: int, session: aiohttp.ClientSession,
                         download_queue: asyncio.Queue, save_queue: asyncio.Queue):
    n_downloaded = 0
    while n_downloaded < limit:
        image_urls = await download_queue.get()
        if image_urls is None:
            break

        while len(image_urls) > 0 and n_downloaded < limit:
            image_url = image_urls.pop()
            if image_url is None:
                continue
            else:
                image = await session.get(image_url)
                save_queue.put_nowait(image)
                n_downloaded += 1

    save_queue.put_nowait(None)
    download_queue.task_done()
    save_queue.task_done()


async def save_image(output_dir: str, save_queue: asyncio.Queue):
    while True:
        image = await save_queue.get()
        if image is None:
            break

        filename = os.path.basename(image.url.path)
        if filename == "New-Product-Placeholder-Default.jpg":
            print("Skipping sneaker with no image...")
            continue
        else:
            filepath = os.path.join(output_dir, filename)

            async with aiofiles.open(filepath, "wb") as f:
                print(f"Saving {filename}...")
                await f.write(await image.read())


if __name__ == "__main__":
    # todo cmd args?
    asyncio.run(start_download())
