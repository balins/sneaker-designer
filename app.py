import argparse
import sys
from pathlib import Path

from utils.scraper import Scraper
from generator.gan import gan
from generator.ddpm import ddpm


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Application mode")

    parser_download = subparsers.add_parser(
        "download", help="Download training dataset")
    parser_download.add_argument("-l", "--limit", dest="limit", nargs="?", default=sys.maxsize, type=int,
                                 help="Limit the number of images to download")
    parser_download.set_defaults(func=download_samples)

    parser_gan = subparsers.add_parser("gan", help="Train GAN")
    parser_gan.add_argument("-e", "--epochs", dest="epochs", required=True, type=int, help="Set the number of "
                            "epochs for training")
    parser_gan.add_argument("--gfrom", dest="G_from",
                            help="Load existing generator from a path")
    parser_gan.add_argument("--dfrom", dest="D_from",
                            help="Load existing discriminator from a path")
    parser_gan.set_defaults(func=run_gan)

    parser_ddpm = subparsers.add_parser("ddpm", help="Train DDPM")
    parser_ddpm.add_argument("-e", "--epochs", dest="epochs", required=True, type=int, help="Set the number of "
                             "epochs for training")
    parser_ddpm.add_argument("--from", dest="load_from",
                             help="Load existing UNet from a path")
    parser_ddpm.set_defaults(func=run_ddpm)

    return parser.parse_args()


def download_samples(limit=sys.maxsize):
    Scraper.run(limit=limit)

# --gfrom C:\Users\balin\Documents\sneaker-designer\gan\models\0114_1554\g105.pt
# --dfrom C:\Users\balin\Documents\sneaker-designer\gan\models\0114_1554\d105.pt


def run_gan(epochs, G_from=None, D_from=None):
    root = Path(__file__).parent / "images" / "train"

    gan.start_training(img_root=root, num_epochs=epochs,
                       G_from=G_from, D_from=D_from)


def run_ddpm(epochs, load_from=None):
    root = Path(__file__).parent / "images" / "train"

    ddpm.start_training(img_root=root, num_epochs=epochs, load_from=load_from)


if __name__ == "__main__":
    args = parse_args()
    kwargs = vars(args)
    func = kwargs.pop("func")
    func(**kwargs)
