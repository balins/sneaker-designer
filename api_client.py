import argparse
import asyncio
import itertools
import logging
import os
import sys
from string import Template

import aiofiles
import aiohttp

_SNEAKERS_PER_PAGE = 100
_API_ENDPOINT = Template(f"https://api.thesneakerdatabase.com/v1/sneakers?limit={_SNEAKERS_PER_PAGE}&page=$page")
_LOG_LEVELS = logging._nameToLevel

_img_size_args = {
    **dict.fromkeys(("s", "small"), "thumbUrl"),
    **dict.fromkeys(("medium", "m"), "imageUrl"),
    **dict.fromkeys(("large", "l"), "smallImageUrl")
}


class _Color:
    RESET = "\u001b[0m"
    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    MAGENTA = "\u001b[35m"
    CYAN = "\u001b[36m"
    GRAY = "\u001b[8m"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", dest="directory", nargs="?", default="images",
                        help="Directory where to save downloaded images")
    parser.add_argument("--limit", dest="limit", nargs="?", default=sys.maxsize, type=int,
                        help="Limit the number of images to download")
    parser.add_argument("--from", dest="page", nargs="?", default=0, type=int,
                        help="Specify a starting page for API json query (defaults to 0)")
    parser.add_argument("--size", dest="image_size", nargs="?", default="s",
                        help=f"Pick the size of downloaded images from {list(_img_size_args.keys())} (defaults to 's')")
    parser.add_argument("--log", dest="log_level", nargs="?", default="INFO",
                        help=f"Pick log level from {list(_LOG_LEVELS.keys())} (defaults to 'INFO')")
    _args = parser.parse_args()

    try:
        size_arg = _img_size_args[_args.image_size]
    except IndexError:
        raise ValueError(f"Argument size has to be in {list(_img_size_args.keys())}! Got '{_args.image_size}' instead.")
    if _args.page < 0 or _args.limit < 0:
        raise ValueError("Starting page number and limit must not be negative!")
    if _args.log_level not in _LOG_LEVELS.keys():
        raise ValueError(f"Log level has to be in {list(_LOG_LEVELS.keys())}! Got '{_args.log_level}' instead.")
    os.makedirs(_args.directory, exist_ok=True)

    return dict(directory=_args.directory, page=_args.page, limit=_args.limit,
                size_arg=size_arg, log_level=_args.log_level)


async def start_bulk_download(output_dir: str, limit=sys.maxsize, starting_page=0, size_arg="thumbUrl"):
    download_queue, save_queue = asyncio.Queue(), asyncio.Queue()

    logging.info("%sStarting download..." % _Color.GREEN)

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.Task(_url_fetcher(limit, size_arg, starting_page, session, download_queue)),
            asyncio.Task(_image_fetcher(session, download_queue, save_queue)),
            asyncio.Task(_image_writer(output_dir, save_queue))
        ]
        await asyncio.gather(*tasks)

    logging.info("%sAll jobs complete!" % _Color.GREEN)


async def _url_fetcher(limit: int, size_arg: str, starting_page: int,
                       session: aiohttp.ClientSession, download_queue: asyncio.Queue):
    n_fetched = 0
    page = itertools.count(starting_page)

    while n_fetched < limit:
        resp = await session.get(_API_ENDPOINT.substitute(page=next(page)))
        sneakers = (await resp.json())["results"]
        if len(sneakers) == 0:
            logging.warning("%sAPI ran out of sneakers! Cannot fetch more images." % _Color.RED)
            break

        image_urls = [sneaker["media"][size_arg]
                      for sneaker in sneakers[:min(_SNEAKERS_PER_PAGE, limit - n_fetched)]
                      if sneaker["media"][size_arg] is not None]

        for image_url in image_urls:
            download_queue.put_nowait(image_url)
            n_fetched += 1

    download_queue.put_nowait(None)
    logging.info("%sFetching urls complete!" % _Color.CYAN)


async def _image_fetcher(session: aiohttp.ClientSession, download_queue: asyncio.Queue, save_queue: asyncio.Queue):
    while True:
        image_url = await download_queue.get()
        if image_url is None:
            break

        image = await session.get(image_url)
        save_queue.put_nowait(image)

    download_queue.task_done()
    save_queue.put_nowait(None)
    logging.info("%sFetching images complete!" % _Color.CYAN)


async def _image_writer(output_dir: str, save_queue: asyncio.Queue):
    while True:
        image = await save_queue.get()
        if image is None:
            break

        filename = os.path.basename(image.url.path)
        filepath = os.path.join(output_dir, filename)

        async with aiofiles.open(filepath, "wb") as f:
            logging.debug("%sSaving %s..." % (_Color.MAGENTA, filename))
            await f.write(await image.read())

    save_queue.task_done()
    logging.info("%sSaving images complete!" % _Color.CYAN)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(format=f"{_Color.GRAY}[%(asctime)s %(levelname)s]{_Color.RESET} %(message)s{_Color.RESET}",
                        level=_LOG_LEVELS[args["log_level"]], datefmt="%H:%M:%S")
    asyncio.run(start_bulk_download(args["directory"], args["limit"], args["page"], args["size_arg"]))
