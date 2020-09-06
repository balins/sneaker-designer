import argparse
import asyncio
import sys
from pathlib import Path

import gan
from api_client import ApiClient


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Application mode")

    parser_download = subparsers.add_parser("download", help="Download training dataset")
    parser_download.add_argument("-l", "--limit", dest="limit", nargs="?", default=sys.maxsize, type=int,
                                 help="Limit the number of images to download")
    parser_download.add_argument("-p", "--page", dest="starting_page", nargs="?", default=0, type=int,
                                 help="Specify a starting page for API json query (defaults to 0)")
    parser_download.add_argument("-s", "--size", dest="image_size", nargs="?", default="s",
                                 help=f"Pick the size of downloaded images from: 's' (default), 'm', 'l'")
    parser_download.set_defaults(func=download_samples)

    parser_train = subparsers.add_parser("train", help="Train GAN")
    parser_train.add_argument("-t", "--test", action="store_true", help="Run training on smaller dataset")
    parser_train.add_argument("-e", "--epochs", dest="epochs", required=True, type=int, help="Set the number of "
                                                                                             "epochs for training")
    parser_train.add_argument("-b", "--batch_size", dest="batch_size", nargs="?", default=256, type=int,
                              help="Set batch size for training (defaults to 256)")
    parser_train.add_argument("-l", "--learning_rate", dest="learning_rate", nargs="?", default=2e-4, type=float,
                              help="Set learning rate for optimizers (defaults to 0.0002)")
    parser_train.add_argument("--beta1", dest="beta1", nargs="?", default=0.5, type=float,
                              help="Set beta1 parameter for optimizers (defaults to 0.5)")
    parser_train.add_argument("--gfrom", dest="G_from", help="Load existing generator from path")
    parser_train.add_argument("--dfrom", dest="D_from", help="Load existing discriminator from path")
    parser_train.set_defaults(func=train)

    return parser.parse_args()


def download_samples(starting_page=0, limit=sys.maxsize, image_size="s"):
    api_client = ApiClient(starting_page=starting_page, limit=limit, image_size=image_size)
    asyncio.run(api_client.start_bulk_download())


def train(epochs, batch_size=256, learning_rate=2e-4, beta1=0.5, G_from=None, D_from=None, test=False):
    if test:
        img_root = Path("images") / "test" / "s"
    else:
        img_root = Path("images") / "training" / "s"

    gan.start_training(img_root=img_root, num_epochs=epochs,
                       batch_size=batch_size, learning_rate=learning_rate,
                       beta1=beta1, G_from=G_from, D_from=D_from)


if __name__ == "__main__":
    args = parse_args()
    kwargs = vars(args)
    func = kwargs.pop("func")
    func(**kwargs)
