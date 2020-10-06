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
from .nets import Discriminator
from .nets import Generator

# built with https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

__log = AppLogger(__name__)
__image_size = 256

ngpu = torch.cuda.device_count()
__uses_cuda = torch.cuda.is_available() and ngpu > 0
device = torch.device("cuda:0" if __uses_cuda else "cpu")

__session_start = datetime.now().strftime("%m%d_%H%M")
models_dir = Path(__file__).parent.parent / Path("models") / __session_start
plots_dir = Path(__file__).parent.parent / Path("plots") / __session_start


def start_training(img_root, num_epochs, batch_size=128, learning_rate=2e-4,
                   beta1=0.5, nz=128, ngf=128, ndf=128, G_from=None, D_from=None):
    netG = Generator(nz=nz, ngf=ngf, ngpu=ngpu).to(device)
    netD = Discriminator(ndf=ndf, ngpu=ngpu).to(device)
    if device.type == "cuda" and ngpu > 1:
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netG.apply(_weights_init)
    netD.apply(_weights_init)

    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    epochG, epochD = 0, 0
    if G_from is not None:
        epochG, netG, optimizerG = _load_state(netG, optimizerG, G_from)
    if D_from is not None:
        epochD, netD, optimizerD = _load_state(netD, optimizerD, D_from)

    dataloader = _get_dataloader(img_root, batch_size)
    training_loop(dataloader, num_epochs,
                  netG, optimizerG, nz, netD, optimizerD,
                  criterion=nn.BCELoss(),
                  g_initial_epoch=epochG, d_initial_epoch=epochD)


def training_loop(dataloader, num_epochs, netG, optimizerG, nz, netD, optimizerD,
                  criterion, g_initial_epoch=0, d_initial_epoch=0):
    def save_status(suffix):
        _save_model(g_initial_epoch+epoch, netG.state_dict(), optimizerG.state_dict(), G_losses[-1],
                    f"generator_{suffix}.pt")
        _save_model(d_initial_epoch+epoch, netD.state_dict(), optimizerD.state_dict(), D_losses[-1],
                    f"discriminator_{suffix}.pt")
        with torch.no_grad():
            fake_ = netG(fixed_noise).detach().cpu()

        generated_fakes = vutils.make_grid(fake_, padding=1, normalize=True)
        _save_plots(generated_fakes, G_losses, D_losses, suffix=suffix)

    __log.info(f"Using {ngpu if __uses_cuda else 0} GPU(s).")
    models_dir.mkdir(exist_ok=True, parents=True)
    real_label = 1.
    fake_label = 0.
    G_losses, D_losses = [], []
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    iters = 0
    epoch = 0

    __log.info("Starting Training Loop...")
    try:
        while epoch < num_epochs:
            for i, data in enumerate(dataloader, 0):
                netD.zero_grad()

                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                netG.zero_grad()

                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if i % 100 == 0:
                    print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                          % (epoch, num_epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                if iters > 0 and iters % 1000 == 0:
                    __log.info(f"Saving intermediate results after {iters} iteration...")
                    save_status(suffix=iters)

                iters += 1
            epoch += 1
    except Exception as e:
        __log.error("Exception occured!")
        __log.error(e, exc_info=True)
    finally:
        __log.info(f"Saving final results...")
        save_status(suffix="final")


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
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    __log.debug(f"Loaded model state from epoch {epoch}, loss {loss}.")

    return epoch, model, optimizer


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
                                   transforms.Pad(padding=(0, 86), fill=256),
                                   transforms.Resize(__image_size),
                                   transforms.CenterCrop(__image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def _save_plots(fake_images, G_losses, D_losses, suffix):
    plots_dir.mkdir(exist_ok=True, parents=True)

    losses_output_path = plots_dir / f"losses_{suffix}.png"
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(fname=losses_output_path)
    plt.close()

    generated_output_path = plots_dir / f"generated_{suffix}.png"
    plt.figure(figsize=(30, 30))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(fake_images, (1, 2, 0)))
    plt.savefig(fname=generated_output_path)
    plt.close()
