import glob
import json
import sys

import requests

from pathlib import Path
from .logger import Logger

logger = Logger(__name__)
SCRAPE_URLS_PATH = Path(__file__).parent / "scrape_urls"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "images" / "train" / "sneakers"


class Scraper:
    @staticmethod
    def run(limit=sys.maxsize, output_dir=DEFAULT_OUTPUT_DIR):
        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Starting download...")
        n_fetched = 0

        for input_file in glob.glob(SCRAPE_URLS_PATH / "*.json"):
            logger.info(f"Opening input dataset {input_file}...")

            with open(input_file, encoding='utf-8') as dataset:
                for page in json.load(dataset)["pages"]:
                    sneakers = page["Products"]

                    for sneaker in sneakers[:min(len(sneakers), limit)]:
                        url = sneaker["media"]["imageUrl"].split('?')[0]

                        logger.debug("Downloading image...")
                        image = requests.get(url).content

                        filepath = output_dir / url.split("/")[-1]

                        with open(filepath, "wb") as out_file:
                            logger.debug(f"Saving {filepath.name}...")
                            out_file.write(image)

                        n_fetched += 1

                        if n_fetched > limit:
                            return n_fetched

        return n_fetched
