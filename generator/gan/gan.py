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

from .generator import Generator
from .discriminator import Discriminator
import utils

seed = 42
# seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)

logger = utils.logger.Logger(__name__)

batch_size = 128
# Size in pixels that the images should be resized to during training
image_size = 64
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Learning rate for optimizers
learning_rate = 2e-4
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable!")

gpu = torch.device("cuda:0")
dataloader_workers = 12

session_start_ts = datetime.now().strftime("%m%d_%H%M")
models_dir = Path(__file__).parent.parent.parent / Path("gan") / \
    Path("models") / session_start_ts
plots_dir = Path(__file__).parent.parent.parent / Path("gan") / \
    Path("plots") / session_start_ts


def start_training(img_root, num_epochs, G_from=None, D_from=None):
    G_net = Generator(nz=nz, ngf=ngf).to(gpu).apply(_weights_init)
    G_net = nn.DataParallel(G_net, [gpu])
    G_optimizer = optim.Adam(
        G_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    G_epoch = 0

    D_net = Discriminator(ndf=ndf).to(gpu).apply(_weights_init)
    D_optimizer = optim.Adam(
        D_net.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    D_net = nn.DataParallel(D_net, [gpu])
    D_epoch = 0

    if G_from is not None:
        G_epoch = utils.model.restore(G_net, G_optimizer, G_from)

    if D_from is not None:
        D_epoch = utils.model.restore(D_net, D_optimizer, D_from)

    criterion = nn.BCELoss().to(gpu)
    dataloader = utils.dataset.dataloader(img_root, batch_size)
    initial_epoch = max(G_epoch, D_epoch)

    _training_loop(dataloader, num_epochs,
                   G_net, G_optimizer, D_net, D_optimizer,
                   criterion=criterion,
                   initial_epoch=initial_epoch)


def _training_loop(dataloader, num_epochs, G_net, G_optimizer, D_net, D_optimizer, criterion, initial_epoch):
    def save_status():
        utils.model.save(G_net, G_optimizer, epoch,
                         G_losses[-1], models_dir / f"g{epoch}.pt")
        utils.model.save(D_net, D_optimizer, epoch,
                         D_losses[-1], models_dir / f"d{epoch}.pt")

        with torch.no_grad():
            fake = G_net(fixed_noise).detach().cpu()

        imgs.append(vutils.make_grid(fake, padding=1, normalize=True))
        _save_plots(imgs, G_losses, D_losses, epoch)

    _save_sample_batch(dataloader)

    models_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)

    real_label = 1.
    fake_label = 0.
    G_losses, D_losses = [], []
    imgs = []
    fixed_noise = torch.randn(64, nz, 1, 1, device=gpu)
    epoch = initial_epoch
    last_epoch = initial_epoch + num_epochs

    logger.info("Starting the training loop...")

    try:
        while epoch < last_epoch:
            for iter, data in enumerate(dataloader, 0):
                # Flip labels with probability 0.005 (every ~200 iterations) to confuse discriminator
                should_invert_labels = random.random() < 0.005

                ############################
                # (1) Update Discrimiator network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                D_net.zero_grad()

                # Format batch
                real = data[0].to(gpu)
                b_size = real.size(0)

                # Add a bit of random noise to the real images
                # d_noise = 0.1 * (torch.randn(real.size(), device=gpu)
                #                  ) * ((last_epoch-epoch) / last_epoch)
                # real = torch.clamp(real + d_noise, min=-1., max=1.)

                if should_invert_labels:
                    probably_real_label = fake_label
                else:
                    probably_real_label = real_label

                label = torch.full((b_size,), probably_real_label,
                                   dtype=torch.float, device=gpu)

                # Forward pass real batch through D
                output = D_net(real).view(-1)

                # Calculate loss on all-real batch
                D_err_real = criterion(output, label)

                # Calculate gradients for D in backward pass
                D_err_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=gpu)
                # Generate fake image batch with G
                fake = G_net(noise)

                if should_invert_labels:
                    probably_fake_label = real_label
                else:
                    probably_fake_label = fake_label

                label.fill_(probably_fake_label)
                # Classify all fake batch with D
                output = D_net(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                D_err_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated with previous gradients
                D_err_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                D_err = D_err_real + D_err_fake
                # Update D
                D_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                G_net.zero_grad()
                # fake labels are real for generator cost
                label.fill_(real_label)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = D_net(fake).view(-1)
                # Calculate G's loss based on this output
                G_err = criterion(output, label)
                # Calculate gradients for G
                G_err.backward()
                D_G_z2 = output.mean().item()
                # Update G
                G_optimizer.step()

                if iter % 50 == 0:
                    logger.info("[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                                % (epoch, last_epoch, D_err.item(), G_err.item(), D_x, D_G_z1, D_G_z2))

                G_losses.append(G_err.item())
                D_losses.append(D_err.item())

            epoch += 1

            if epoch % 5 == 0:
                logger.info(
                    f"Saving intermediate results after {epoch} epochs...")
                save_status()

    except Exception as e:
        logger.error("An exception occured!")
        logger.error(e, exc_info=True)
    finally:
        logger.info(f"Saving final results...")
        save_status()


def _weights_init(net):
    # custom weights initialization called on the generator and discriminator
    classname = net.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)


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


def _save_plots(fake_images, G_losses, D_losses, epoch):
    plots_dir.mkdir(exist_ok=True, parents=True)

    # save plot
    losses_output_path = plots_dir / f"losses_{epoch}.png"
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
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
