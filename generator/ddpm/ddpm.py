import torch
import torch.nn.functional as F
from .unet import UNet
from datetime import datetime
from pathlib import Path

import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import matplotlib.animation as animation
import utils

seed = 42
# seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)

logger = utils.logger.Logger(__name__)

# Define beta schedule
T = 300
schedule_start = 1e-4
schedule_end = 1e-2
betas = torch.from_numpy(np.geomspace(schedule_start, schedule_end, T, dtype=np.float32))
# betas = torch.linspace(schedule_start, schedule_end, T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
sqrt_recip_alphas = torch.sqrt(1. / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

batch_size = 128
learning_rate = 1e-4

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable!")

gpu = torch.device("cuda:0")
dataloader_workers = 12

session_start_ts = datetime.now().strftime("%m%d_%H%M")
models_dir = Path(__file__).parent.parent.parent / Path("ddpm") / \
    Path("models") / session_start_ts
plots_dir = Path(__file__).parent.parent.parent / Path("ddpm") / \
    Path("plots") / session_start_ts


def start_training(img_root, num_epochs, load_from=None):
    model = UNet().to(gpu)
    model = nn.DataParallel(model, [gpu])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    initial_epoch = 0

    if load_from is not None:
        initial_epoch = utils.model.restore(model, optimizer, load_from)

    criterion = nn.L1Loss().to(gpu)
    dataloader = utils.dataset.dataloader(img_root, batch_size, drop_last=True)

    _training_loop(dataloader, num_epochs, model, optimizer,
                   criterion=criterion, initial_epoch=initial_epoch)


def _training_loop(dataloader, num_epochs, model, optimizer, criterion, initial_epoch):
    def save_status():
        utils.model.save(model, optimizer, epoch,
                         losses[-1], models_dir / f"{epoch}.pt")

        with torch.no_grad():
            fake = sample_images(model, fixed_noise).detach().cpu()

        imgs.append(vutils.make_grid(fake, padding=1, normalize=True))
        _save_plots(imgs, losses, epoch)

    #_save_sample_batch(dataloader)

    models_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)

    losses = []
    imgs = []
    fixed_noise = torch.randn((64, 3, 64, 64), device=gpu)
    epoch = initial_epoch
    last_epoch = initial_epoch + num_epochs

    logger.info("Starting the training loop...")

    try:
        while epoch < last_epoch:
            for iter, data in enumerate(dataloader, 0):
                optimizer.zero_grad()

                t = torch.randint(0, T, (batch_size,), device=gpu).long()
                x_noisy, noise = forward_diffusion_sample(data[0], t, gpu)
                noise_pred = model(x_noisy, t)

                err = criterion(noise_pred, noise)
                err.backward()
                optimizer.step()

                if iter % 50 == 0:
                    logger.info("[%d/%d]\tLoss: %.4f" %
                                (epoch, last_epoch, err.item()))

                losses.append(err.item())

            epoch += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Saving intermediate results after {epoch} epochs...")
                save_status()

    except Exception as e:
        logger.error("An exception occured!")
        logger.error(e, exc_info=True)
    finally:
        logger.info(f"Saving final results...")
        save_status()


def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = torch.gather(vals, -1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cuda"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """

    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(
        sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * \
        noise.to(device), noise.to(device)


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if torch.all(t):
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
    else:
        # if any 0 in t, return the images
        return model_mean

@torch.no_grad()
def sample_images(model, fixed_noise):
    imgs = fixed_noise

    for i in range(0,T)[::-1]:
        t = torch.full((imgs.shape[0],), i, device=gpu, dtype=torch.long)
        imgs = sample_timestep(model, imgs, t)

    return imgs


def _save_sample_batch(dataloader):
    logger.info("Saving sample batch...")

    real_batch = next(iter(dataloader))[0].detach().cpu()[:64]
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(
        real_batch, padding=1, normalize=True), (1, 2, 0)))
    plt.savefig(fname="sample_batch.png")
    plt.close()


def _save_plots(fake_images, losses, epoch):
    plots_dir.mkdir(exist_ok=True, parents=True)

    # save plot
    losses_output_path = plots_dir / f"losses_{epoch}.png"
    plt.figure(figsize=(10, 5))
    plt.title("UNet Loss During Training")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(fname=losses_output_path)
    plt.close()

    # save the last grid generated images
    generated_output_path = plots_dir / f"generated_{epoch}.png"
    plt.figure(figsize=(30, 30))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(fake_images[-1], (1, 2, 0)))
    plt.savefig(fname=generated_output_path)
    plt.close()

    # save animation
    animation_output_path = plots_dir / f"animation_{epoch}.html"
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]
           for i in fake_images[:50]]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True)

    with open(animation_output_path, 'w') as file:
        file.write(ani.to_jshtml())
    plt.close()
