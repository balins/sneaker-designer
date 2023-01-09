import argparse
import sys
from pathlib import Path

from generator.gan import gan
from utils.scraper import Scraper


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Application mode")

    parser_download = subparsers.add_parser(
        "download", help="Download training dataset")
    parser_download.add_argument("-l", "--limit", dest="limit", nargs="?", default=sys.maxsize, type=int,
                                 help="Limit the number of images to download")
    parser_download.set_defaults(func=download_samples)

    parser_train = subparsers.add_parser("train", help="Train GAN")
    parser_train.add_argument(
        "-t", "--test", action="store_true", help="Run training on smaller dataset")
    parser_train.add_argument("-e", "--epochs", dest="epochs", required=True, type=int, help="Set the number of "
                                                                                             "epochs for training")
    parser_train.add_argument("--gfrom", dest="G_from",
                              help="Load existing generator from a path")
    parser_train.add_argument("--dfrom", dest="D_from",
                              help="Load existing discriminator from a path")
    parser_train.set_defaults(func=train)

    return parser.parse_args()


def download_samples(limit=sys.maxsize):
    Scraper.run(limit=limit)


def train(epochs, G_from=None, D_from=None, test=False):
    root = Path(__file__).parent / "images"

    if test:
        img_root = root / "test"
    else:
        img_root = root / "train"

    gan.start_training(img_root=img_root, num_epochs=epochs, G_from=G_from, D_from=D_from)


if __name__ == "__main__":
    args = parse_args()
    kwargs = vars(args)
    func = kwargs.pop("func")
    func(**kwargs)
