import asyncio
import itertools
import os
import sys
from string import Template

import aiofiles
import aiohttp

from logger import Logger


class AsyncApiClient:
    __SNEAKERS_PER_PAGE = 100
    __API_ENDPOINT = Template(f"https://api.thesneakerdatabase.com/v1/sneakers?limit={__SNEAKERS_PER_PAGE}&page=$page")
    __IMG_SIZE_ARGS = dict(s="thumbUrl", m="imageUrl", l="smallImageUrl")

    def __init__(self, output_dir: str, limit=sys.maxsize, starting_page=0, image_size="s", log_level="INFO"):
        try:
            size_arg = AsyncApiClient.__IMG_SIZE_ARGS[image_size]
        except IndexError:
            raise ValueError(f"Argument size has to be one from 's' (default), 'm', 'l'! Got '{image_size}' instead.")
        if starting_page < 0 or limit < 0:
            raise ValueError("Starting page number and limit must not be negative!")

        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.limit = limit
        self.starting_page = starting_page
        self.image_size = size_arg
        self.log = Logger("AsyncApiClient")
        self.log.setLevel(log_level)
        self.session = None
        self.download_queue = None
        self.save_queue = None

    async def start_bulk_download(self):
        self.log.info("Starting download...")
        self.download_queue, self.save_queue = asyncio.Queue(), asyncio.Queue()

        tasks = [
            asyncio.Task(self.__url_fetcher()),
            asyncio.Task(self.__image_fetcher()),
            asyncio.Task(self.__image_writer())
        ]

        try:
            self.session = aiohttp.ClientSession()
            await asyncio.gather(*tasks)
        except Exception:
            self.log.error("Unexpected error!")
            raise
        finally:
            self.log.warning("Terminating...")
            await self.session.close()

        self.log.info("All jobs complete!")

    async def __url_fetcher(self):
        n_fetched = 0
        page = itertools.count(self.starting_page)

        while n_fetched < self.limit:
            resp = await self.session.get(self.__API_ENDPOINT.substitute(page=next(page)))
            sneakers = (await resp.json())["results"]
            if len(sneakers) == 0:
                self.log.warning("API ran out of sneakers! Cannot fetch more images.")
                break

            image_urls = [sneaker["media"][self.image_size]
                          for sneaker in sneakers[:min(self.__SNEAKERS_PER_PAGE, self.limit - n_fetched)]
                          if sneaker["media"][self.image_size] is not None]

            for image_url in image_urls:
                self.download_queue.put_nowait(image_url)
                n_fetched += 1

        self.download_queue.put_nowait(None)
        self.log.info("Fetching urls complete!")

    async def __image_fetcher(self):
        while True:
            image_url = await self.download_queue.get()
            if image_url is None:
                break

            image = await self.session.get(image_url)
            self.save_queue.put_nowait(image)

        self.download_queue.task_done()
        self.save_queue.put_nowait(None)
        self.log.info("Fetching images complete!")

    async def __image_writer(self):
        while True:
            image = await self.save_queue.get()
            if image is None:
                break

            filename = os.path.basename(image.url.path)
            filepath = os.path.join(self.output_dir, filename)

            async with aiofiles.open(filepath, "wb") as f:
                self.log.debug(f"Saving {filename}...")
                await f.write(await image.read())

        self.save_queue.task_done()
        self.log.info("Saving images complete!")
