import torch
import torch.nn as nn
from torch.optim import Adam

from module import extract_content, extract_style

import torch.nn.functional as F


def pred_by_features(generator, content_list, style_features):
    latent_vector = torch.cat((content_list[-1], style_features), dim=1)
    out = latent_vector

    for i in range(len(generator.generator)):
        out = generator.generator[i](out)

        if i < 3:
            out += F.interpolate(content_list[-i-1], scale_factor=2.0)

    return out


def interpolation_inverse(
    device,
    generator,
    content_letters,
    style_letters,
    style_features108,
    max_epoch,
    purpose_loss,
    learning_rate=0.01
    ):

    # content_letters shape: (1, 1, 128, 128)
    # style_letters shape: (1, 1, 128, 128)
    # style_features108 shape: (108, 128, 4, 4)

    assert style_letters.size(0) == 1

    generator = generator.to(device)
    generator.eval()

    content_letters = content_letters.to(device)
    style_letters = style_letters.to(device)
    style_features108 = style_features108.to(device)

    with torch.no_grad():
        content_list = extract_content(generator, content_letters)
    
    weights = torch.rand(size=(108, 1, 1, 1)).to(device)
    weights /= torch.sum(weights)
    
    weights.requires_grad = True

    optim = Adam([weights], lr=learning_rate, betas=(0.5, 0.999))

    criterion = F.mse_loss
    for epoch in range(max_epoch):
        style_features = torch.sum(style_features108 * weights, dim=0, keepdim=True)

        pred = pred_by_features(generator, content_list, style_features)

        loss = criterion(pred, style_letters)

        if loss.item() <= purpose_loss:
            break

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    style_features = torch.sum(style_features108 * weights, dim=0, keepdim=True)

    return style_features
