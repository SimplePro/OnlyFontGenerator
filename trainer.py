import pickle

from model import Generator, Loss
from torch.optim import Adam

import matplotlib.pyplot as plt

from tqdm import tqdm



class Trainer:

    def __init__(
                self,
                generator, # 생성 모델
                device, # 디바이스
                trainloader, # 학습 로더
                validloader, # 검증 로더
                lr=0.002, # 학습률
                mse_alpha=1,
                style_alpha=1,
                content_alpha=1,
            ):

        self.device = device

        self.generator = generator.to(device)

        self.bce_criterion = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.gen_optim = Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.trainloader = trainloader
        self.validloader = validloader

        self.mse_alpha = mse_alpha
        self.style_alpha = style_alpha
        self.content_alpha = content_alpha

        self.history = {
                            "train": {
                                "total": [],
                                "mse": [],
                                "style": [],
                                "content": [],
                            },

                            "valid": {
                                "total": [],
                                "mse": [],
                                "style": [],
                                "content": [],
                            }
                        }
        
        self.best_loss = 1e5
        self.best_params = None

        self.input_letters, self.gothic_letters, self.output_letters, _ = self.validloader.get(batch_size=3)

        self.input_letters = self.input_letters.to(device).type(torch.float32)
        self.gothic_letters = self.gothic_letters.to(device).type(torch.float32)
        self.output_letters = self.output_letters.to(device).type(torch.float32)


    @torch.no_grad()
    def save_pred(self, path):
        plt.clf()

        self.generator.eval()

        input_images = (self.input_letters.cpu().detach().numpy() * 255).astype(np.uint)
        gothic_images = (self.gothic_letters.cpu().detach().numpy() * 255).astype(np.uint)
        output_images = (self.output_letters.cpu().detach().numpy() * 255).astype(np.uint)

        pred = self.generator(self.input_letters, self.gothic_letters)
        pred_images = (pred.cpu().detach().numpy() * 255).astype(np.uint)

        for i in range(3):
            input_img = input_images[i].transpose(1, 2, 0).reshape(128, 128)
            gothic_img = gothic_images[i].transpose(1, 2, 0).reshape(128, 128)
            output_img = output_images[i].transpose(1, 2, 0).reshape(128, 128)
            pred_img = pred_images[i].transpose(1, 2, 0).reshape(128, 128)

            plt.subplot(3, 4, 4*i+1)
            plt.imshow(input_img, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 4, 4*i+2)
            plt.imshow(gothic_img, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 4, 4*i+3)
            plt.imshow(output_img, cmap="gray")
            plt.axis("off")

            plt.subplot(3, 4, 4*i+4)
            plt.imshow(pred_img, cmap="gray")
            plt.axis("off")

        plt.savefig(path, dpi=300)


    @torch.no_grad()
    def save_history(self, path):

        plt.clf()

        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(range(len(self.history["train"]["total"])), self.history["train"]["total"], label="train-loss", color="#00687D")
        ax1.legend()

        ax2 = ax1.twiny()
        ax2.plot(range(len(self.history["valid"]["total"])), self.history["valid"]["total"], label="valid-loss", color="#F07D28")
        ax2.legend()

        plt.savefig(path)


    def train(self):
        self.generator.train()

        avg_loss = {
            "total": 0,
            "mse": 0,
            "style": 0,
            "content": 0,
        }

        batch_size = 64

        for i in tqdm(range(len(self.trainloader))):

            input_letters, gothic_letters, output_letters, font_labels = self.trainloader.get()

            input_letters = input_letters.to(self.device).type(torch.float32)
            gothic_letters = gothic_letters.to(self.device).type(torch.float32)
            output_letters = output_letters.to(self.device).type(torch.float32)
            font_labels = font_labels.to(self.device)

            pred = self.generator(input_letters, gothic_letters)

            pred_styles = self.generator.style_extractor(pred)
            pred_content = pred
            for i in range(len(self.generator.content_extractor)):
                pred_content = self.generator.content_extractor[i](pred_content)

            mse_loss = F.mse_loss(pred, output_letters)
            style_loss = F.mse_loss(pred_styles, self.generator.style)
            content_loss = F.mse_loss(pred_content, self.generator.content_list[-1])

            mse_loss *= self.mse_alpha
            style_loss *= self.style_alpha
            content_loss *= self.content_alpha

            total_loss = mse_loss + style_loss + content_loss
            
            self.gen_optim.zero_grad()
            total_loss.backward()
            self.gen_optim.step()

            self.history["train"]["total"].append(total_loss.item())
            self.history["train"]["mse"].append(mse_loss.item())
            self.history["train"]["style"].append(style_loss.item())
            self.history["train"]["content"].append(content_loss.item())

            avg_loss["total"] += total_loss.item()
            avg_loss["mse"] += mse_loss.item()
            avg_loss["style"] += style_loss.item()
            avg_loss["content"] += content_loss.item()
            
        avg_loss["total"] /= len(self.trainloader)
        avg_loss["mse"] /= len(self.trainloader)
        avg_loss["style"] /= len(self.trainloader)
        avg_loss["content"] /= len(self.trainloader)

        return list(avg_loss.values())

    
    @torch.no_grad()
    def valid(self):
        self.generator.eval()

        avg_loss = {
            "total": 0,
            "mse": 0,
            "style": 0,
            "content": 0,
        }

        batch_size = 128

        for i in tqdm(range(len(self.validloader))):

            input_letters, gothic_letters, output_letters, font_labels = self.validloader.get()

            input_letters = input_letters.to(self.device).type(torch.float32)
            gothic_letters = gothic_letters.to(self.device).type(torch.float32)
            output_letters = output_letters.to(self.device).type(torch.float32)
            font_labels = font_labels.to(self.device)

            pred = self.generator(input_letters, gothic_letters)

            pred_styles = self.generator.style_extractor(pred)
            pred_content = pred
            for i in range(len(self.generator.content_extractor)):
                pred_content = self.generator.content_extractor[i](pred_content)

            mse_loss = F.mse_loss(pred, output_letters)
            style_loss = F.mse_loss(pred_styles, self.generator.style)
            content_loss = F.mse_loss(pred_content, self.generator.content_list[-1])

            mse_loss *= self.mse_alpha
            style_loss *= self.style_alpha
            content_loss *= self.content_alpha

            total_loss = mse_loss + style_loss + content_loss

            self.history["valid"]["total"].append(total_loss.item())
            self.history["valid"]["mse"].append(mse_loss.item())
            self.history["valid"]["style"].append(style_loss.item())
            self.history["valid"]["content"].append(content_loss.item())

            avg_loss["total"] += total_loss.item()
            avg_loss["mse"] += mse_loss.item()
            avg_loss["style"] += style_loss.item()
            avg_loss["content"] += content_loss.item()
            
        avg_loss["total"] /= len(self.validloader)
        avg_loss["mse"] /= len(self.validloader)
        avg_loss["style"] /= len(self.validloader)
        avg_loss["content"] /= len(self.validloader)

        if avg_loss["total"] <= self.best_loss:
            self.best_loss = avg_loss["total"]
            self.best_params = self.generator.state_dict()

        return list(avg_loss.values())

    
    def set_alphas(self, mse_alpha, style_alpha, content_alpha):
        self.mse_alpha = mse_alpha
        self.style_alpha = style_alpha
        self.content_alpha = content_alpha


    def run(
            self,
            epochs: tuple,
            history_base_path,
            pred_base_path,
            trainer_base_path=None
        ):
        
        for epoch in range(*epochs):

            print("-" * 50 + f" EPOCH: [{epoch+1}/{epochs[1]}] " + "-" * 50, end="\n\n")
            
            print("TRAIN", end="\n")
            train_total_loss, train_mse_loss, train_style_loss, train_content_loss = self.train()
            print(f"total_loss: {train_total_loss}, mse_loss: {train_mse_loss}, style_loss: {train_style_loss}, content_loss: {train_content_loss}", end="\n\n")

            print("VALID", end="\n")
            valid_total_loss, valid_mse_loss, valid_style_loss, valid_content_loss = self.valid()
            print(f"total_loss: {valid_total_loss}, mse_loss: {valid_mse_loss}, style_loss: {valid_style_loss}, content_loss: {valid_content_loss}", end="\n\n")
            
            self.save_history(f"{history_base_path}/epoch-{epoch+1}.png")
            self.save_pred(f"{pred_base_path}/epoch-{epoch+1}.png")

            if trainer_base_path != None:
                with open(trainer_base_path + "/trainer.pickle", "wb") as f:
                    data = {
                        "model": self.generator.state_dict(), 
                        "best_loss": self.best_loss, 
                        "best_params": self.best_params, 
                        "input letters": self.input_letters, 
                        "gothic letters": self.gothic_letters,
                        "output letters": self.output_letters,
                        "history": self.history,
                        "mse alpha": self.mse_alpha,
                        "style alpha": self.style_alpha,
                        "content alpha": self.content_alpha
                    }
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)