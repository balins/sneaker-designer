from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from app_logger import AppLogger
from nets import Discriminator
from nets import Generator

# source https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

__log = AppLogger(__name__)
__image_size = 64
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

device = torch.device("cuda:0")

__session_start = datetime.now().strftime("%m%d_%H%M")
models_dir = Path("models") / __session_start
plots_dir = Path("plots") / __session_start


def start_training(img_root, num_epochs, batch_size=256, learning_rate=2e-4, beta1=0.5, G_from=None, D_from=None):
    netG = Generator(nz=nz, ngf=ngf).to(device)
    netG.apply(_weights_init)
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    netD = Discriminator(ndf=ndf).to(device)
    netD.apply(_weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    if G_from is not None:
        netG, optimizerG = _load_state(netG, optimizerG, G_from)
    if D_from is not None:
        netD, optimizerD = _load_state(netD, optimizerD, D_from)

    dataloader = _get_dataloader(img_root, batch_size)
    fake_images, G_losses, D_losses = training_loop(dataloader, num_epochs,
                                                    netG, optimizerG,
                                                    netD, optimizerD,
                                                    criterion=nn.BCELoss())

    _save_plots(fake_images, G_losses, D_losses)


def training_loop(dataloader, num_epochs, netG, optimizerG, netD, optimizerD, criterion):
    models_dir.mkdir(exist_ok=True, parents=True)

    real_label = 1.
    fake_label = 0.
    G_losses, D_losses = [], []
    iters = 0

    __log.info("Starting Training Loop...")
    try:
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D"s loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G"s loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if iters % 1000 == 0:
                    __log.info(f"Saving intermediate models after {iters} iteration...")
                    _save_model(num_epochs, netG.state_dict(), optimizerG.state_dict(), G_losses[-1],
                                f"generator_{iters}.pt")
                    _save_model(num_epochs, netD.state_dict(), optimizerD.state_dict(), D_losses[-1],
                                f"discriminator_{iters}.pt")

                iters += 1
    except Exception as e:
        __log.error("Exception occured!")
        __log.error(e, exc_info=True)
    finally:
        __log.info("Saving models before stopping...")
        _save_model(num_epochs, netG.state_dict(), optimizerG.state_dict(), G_losses[-1], "generator_final.pt")
        _save_model(num_epochs, netD.state_dict(), optimizerD.state_dict(), D_losses[-1], "discriminator_final.pt")

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()

    generated_fakes = vutils.make_grid(fake, padding=2, normalize=True)

    return generated_fakes, G_losses, D_losses


def _save_model(epoch, model_state_dict, optimizer_state_dict, loss, output_filename):
    model_output_path = models_dir / output_filename
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "loss": loss,
    }, model_output_path)
    __log.debug(f"Model output path: {model_output_path}")


def _load_state(model, optimizer, state_input_path):
    __log.debug(f"State input path: {state_input_path}")
    checkpoint = torch.load(state_input_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    __log.debug(f"Loaded model state from epoch {epoch}, loss {loss}.")

    return model, optimizer


# custom weights initialization called on netG and netD
def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _get_dataloader(dataroot, batch_size):
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(__image_size),
                                   transforms.CenterCrop(__image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def _save_plots(fake_images, G_losses, D_losses):
    plots_dir.mkdir(exist_ok=True, parents=True)

    losses_output_path = plots_dir / "losses.png"
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(fname=losses_output_path)
    plt.close()

    generated_output_path = plots_dir / "generated.png"
    plt.figure(figsize=(30, 30))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(fake_images, (1, 2, 0)))
    plt.savefig(fname=generated_output_path)
