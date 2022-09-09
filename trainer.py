import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
import torch.nn.functional as F

import wandb

import pickle

from argparse import ArgumentParser

from models import VGGLoss


class Trainer:

    def __init__(
                self,
                device,
                generator, # 생성 모델
                trainloader, # 학습 로더
                validloader, # 검증 로더
                mse_penalty,
                lr=0.002, # 학습률
                could_wandb=False,
            ):

        self.device = device

        self.generator = generator.to(self.device)

        self.mse_criterion = F.mse_loss

        self.gen_optim = Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.trainloader = trainloader
        self.validloader = validloader

        self.mse_penalty = mse_penalty
        
        self.best_loss = 1e5
        self.best_params = None

        self.test_content_letters, self.test_style_letters, self.test_style_labels = self.validloader.get(batch_size=25)

        self.test_content_letters = self.test_content_letters.to(self.device).type(torch.float32)
        self.test_style_letters = self.test_style_letters.to(self.device).type(torch.float32)
        self.test_style_labels = self.test_style_labels.to(self.device).type(torch.float32)

        self.train_loss = []

        self.valid_loss = []

        self.test_images = []

        self.could_wandb = could_wandb


    @torch.no_grad()
    def get_pred_image(self):
        self.generator.eval()

        pred = self.generator(self.test_content_letters, self.test_style_labels)
        grid = make_grid(pred.detach().cpu(), nrow=5, normalize=True)
        image = transforms.ToPILImage()(grid)
        
        return image


    def train(self):

        self.generator.train()

        avg_loss = 0

        for _ in range(len(self.trainloader)):

            content_letters, style_letters, style_labels = self.trainloader.get()
            content_letters = content_letters.to(self.device).type(torch.float32)
            style_letters = style_letters.to(self.device).type(torch.float32)
            style_labels = style_labels.to(self.device).type(torch.float32)

            generated_image = self.generator(content_letters, style_labels)

            loss = self.mse_criterion(generated_image, style_letters)

            loss = self.mse_penalty * loss

            avg_loss += loss.item()

            self.gen_optim.zero_grad()
            loss.backward()
            self.gen_optim.step()

            if self.could_wandb:
                wandb.log({"train_loss": loss.item()})

            self.train_loss.append(loss.item())

        avg_loss /= len(self.trainloader)

        return avg_loss
    

    @torch.no_grad()
    def valid(self):

        self.generator.eval()

        avg_loss = 0

        for _ in range(len(self.validloader)):

            content_letters, style_letters, style_labels = self.validloader.get()
            content_letters = content_letters.to(self.device).type(torch.float32)
            style_letters = style_letters.to(self.device).type(torch.float32)
            style_labels = style_labels.to(self.device).type(torch.float32)

            generated_image = self.generator(content_letters, style_labels)

            loss = self.mse_criterion(generated_image, style_letters)

            loss = self.mse_penalty * loss

            avg_loss += loss.item()

            if self.could_wandb:
                wandb.log({"valid_loss": loss.item()})

            self.valid_loss.append(loss.item())

        avg_loss /= len(self.validloader)

        if avg_loss <= self.best_loss:
            self.best_loss = avg_loss
            self.best_params = self.generator.state_dict()

        return avg_loss


    def load_trainer(self, path, wandb_log=True):
        with open(path, 'rb') as f:
            trainer_data = pickle.load(f)

        self.generator.load_state_dict(trainer_data["generator_params"])
        self.best_loss = trainer_data["best_loss"]
        self.best_params = trainer_data["best_params"]
        self.test_content_letters = trainer_data["test_content_letters"]
        self.test_style_letters = trainer_data["test_style_letters"]
        self.test_style_labels = trainer_data["test_style_labels"]
        self.test_images = trainer_data["test_images"]
        self.train_loss = trainer_data["train_loss"]
        self.valid_loss = trainer_data["valid_loss"]

        if wandb_log and self.could_wandb:
            wandb.config.update({"best_loss": self.best_loss})

            for loss in self.train_loss:
                wandb.log({"train_loss": loss})

            for loss in self.valid_loss:
                wandb.log({"valid_loss": loss})

            for i, img in enumerate(self.test_images):
                wandb.log({"pred_image": wandb.Image(img)})


    def run(self, epochs: tuple, trainer_path: str):
        
        for epoch in range(*epochs):

            print("-" * 50 + f" EPOCH: [{epoch+1}/{epochs[1]}] " + "-" * 50, end="\n\n")
            
            print("TRAIN", end="\n")
            train_loss = self.train()
            print(f"train_loss: {train_loss}", end="\n\n")

            print("VALID", end="\n")
            valid_loss = self.valid()
            print(f"valid_loss: {valid_loss}", end="\n\n")

            pred_image = self.get_pred_image()
            if self.could_wandb:
                wandb.log({"pred_image": wandb.Image(pred_image)})
            self.test_images.append(pred_image)

            trainer_data = {
                "generator_params": self.generator.state_dict(),
                "best_loss": self.best_loss,
                "best_params": self.best_params,
                "test_content_letters": self.test_content_letters,
                "test_style_letters": self.test_style_letters,
                "test_style_labels": self.test_style_labels,
                "test_images": self.test_images,
                "train_loss": self.train_loss,
                "valid_loss": self.valid_loss
            }

            with open(trainer_path, "wb") as f:
                pickle.dump(trainer_data, f, pickle.HIGHEST_PROTOCOL)

            if self.could_wandb:
                wandb.config.update({"best_loss": self.best_loss})

                

if __name__ == '__main__':
    
    from models.generator3 import Generator

    from utils import Font, FontDataset, FontDataLoader

    # wandb.init(project="OnlyFontGenerator", entity="donghwankim")

    parser = ArgumentParser(description="OnlyFontGenerator hyper parameters")

    parser.add_argument("--learning_rate", '-l', default=0.0008, type=float, help="learning rate of Generator and Discriminator")

    parser.add_argument("--train_batch_size", default=16, help="batch size that used while model train")

    parser.add_argument("--valid_batch_size", default=16*2, help="batch size that used while model evaluate (valid)")

    parser.add_argument("--mse_penalty", default=5, help="constant that will multiply with mse_loss")

    parser.add_argument("--epochs", "-e", default=(0, 10), type=tuple, help="epochs that type is tuple")

    args = parser.parse_args()

    # wandb.config.update(args)


    with open("./data/train_dataset_list.pickle", "rb") as f:
        train_dataset_list = pickle.load(f)
    
    with open("./data/valid_dataset_list.pickle", "rb") as f:
        valid_dataset_list = pickle.load(f)

    trainloader = FontDataLoader(font_dataset_list=train_dataset_list, batch_size=args.train_batch_size)
    validloader = FontDataLoader(font_dataset_list=valid_dataset_list, batch_size=args.valid_batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    generator = Generator()

    trainer = Trainer(
        device=device,
        generator=generator,
        trainloader=trainloader,
        validloader=validloader,
        mse_penalty=args.mse_penalty,
        lr=args.learning_rate,
        could_wandb=False
    )
    
    trainer.run(epochs=(0, 1), trainer_path="./trainer.pickle")
