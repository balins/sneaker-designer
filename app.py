import argparse
import asyncio
import sys

from api_client import AsyncApiClient


def download_samples():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="directory", nargs="?", default="images",
                        help="Directory where to save downloaded images")
    parser.add_argument("-l", "--limit", dest="limit", nargs="?", default=sys.maxsize, type=int,
                        help="Limit the number of images to download")
    parser.add_argument("-p", "--page", dest="page", nargs="?", default=0, type=int,
                        help="Specify a starting page for API json query (defaults to 0)")
    parser.add_argument("-s", "--size", dest="image_size", nargs="?", default="s",
                        help=f"Pick the size of downloaded images from: 's' (default), 'm', 'l'")
    parser.add_argument("-L", "--log", dest="log_level", nargs="?", default="INFO",
                        help=f"Pick log level from the Python Standard Library 'logging' module (defaults to 'INFO')")
    args = parser.parse_args()

    api_client = AsyncApiClient(output_dir=args.directory, starting_page=args.page, limit=args.limit,
                                image_size=args.image_size, log_level=args.log_level)
    asyncio.run(api_client.start_bulk_download())


def train():
    pass


if __name__ == "__main__":
    download_samples()
    train()
