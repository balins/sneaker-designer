import torch
from .logger import Logger

logger = Logger(__name__)


def save(net, optimizer, epoch, loss, model_output_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, model_output_path)
    logger.info(f"Model output path: {model_output_path}")


def restore(net, optimizer, model_input_path):
    logger.info(f"Model input path: {model_input_path}")
    checkpoint = torch.load(model_input_path)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    logger.info(f"Loaded model state from epoch {epoch}, loss {loss}.")

    return epoch
