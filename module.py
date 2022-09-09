import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image



def extract_content(generator, content_letters, could_adain=False):
    content_list = [content_letters]

    for i in range(len(generator.content_extractor)):
        content_list.append(generator.content_extractor[i](content_list[-1]))

    return content_list


def extract_style(generator, style_labels):
    style = generator.style_extractor(style_labels)

    return style


@torch.no_grad()
def show_pred(device, generator, dataloader, n=10):

    content_letters, style_letters, style_labels = dataloader.get(batch_size=n)

    content_letters = content_letters.to(device).type(torch.float32)
    style_letters = style_letters.to(device).type(torch.float32)
    style_labels = style_labels.to(device).type(torch.float32)

    content_images = (content_letters.cpu().detach().numpy() * 255).astype(np.uint)
    style_images = (style_letters.cpu().detach().numpy() * 255).astype(np.uint)

    pred = generator(content_letters, style_labels)
    pred_images = (pred.cpu().detach().numpy() * 255).astype(np.uint)

    plt.figure(figsize=(n, 3*n))

    for i in range(n):

        content_img = content_images[i].transpose(1, 2, 0).reshape(128, 128)
        style_img = style_images[i].transpose(1, 2, 0).reshape(128, 128)
        pred_img = pred_images[i].transpose(1, 2, 0).reshape(128, 128)

        plt.subplot(n, 3, 3*i+1)
        plt.imshow(content_img, cmap="gray")
        plt.axis("off")

        plt.subplot(n, 3, 3*i+2)
        plt.imshow(style_img, cmap="gray")
        plt.axis("off")

        plt.subplot(n, 3, 3*i+3)
        plt.imshow(pred_img, cmap="gray")
        plt.axis("off")


@torch.no_grad()
def pred_by_style_content(generator, content_list, style_features):
    latent_vector = torch.cat((content_list[-1], style_features), dim=1)
    out = latent_vector

    for i in range(len(generator.generator)):
        out = generator.generator[i](out)

        if i < 3:
            out += F.interpolate(content_list[-i-1], scale_factor=2.0)

    return out


@torch.no_grad()
def interpolate(generator, save_dir, content_list, start_style_features, end_style_features, step_n=100):
    step_gap = (end_style_features - start_style_features) / step_n

    style = start_style_features

    for step in range(step_n):
        pred = pred_by_style_content(generator, content_list, style) # (1, 1, 128, 128)
        pred_img = transforms.ToPILImage()(pred.cpu().detach().squeeze(0))
        
        pred_img.save(f"{save_dir}/step{step}.png")

        style += step_gap


def make_gif(paths, save_path, fps=500):
    img, *imgs = [Image.open(path) for path in paths]
    img.save(fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=fps, loop=1)
