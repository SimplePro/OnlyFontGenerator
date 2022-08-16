import pickle

from model import Generator, Loss
from torch.optim import Adam

import matplotlib.pyplot as plt

from tqdm import tqdm



class Trainer:

    def __init__(
                self,
                generator, # 생성 모델
                discriminator, # 판별 모델
                device, # 디바이스
                trainloader, # 학습 로더
                validloader, # 검증 로더
                lr=0.002, # 학습률
                mse_alpha=1,
                style_alpha=1,
                content_alpha=1,
                style_class_alpha=1,
                discriminate_alpha=1
            ):

        self.device = device

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.bce_criterion = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.gen_optim = Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.dis_optim = Adam(self.discriminator.parameters(), lr=lr*2, betas=(0.5, 0.999))

        self.trainloader = trainloader
        self.validloader = validloader

        self.mse_alpha = mse_alpha
        self.style_alpha = style_alpha
        self.content_alpha = content_alpha
        self.style_class_alpha = style_class_alpha
        self.discriminate_alpha = discriminate_alpha

        self.history = {
                            "train": {
                                "total": [],
                                "mse": [],
                                "style": [],
                                "content": [],
                                "style class": [],
                                "discriminate": []
                            },

                            "valid": {
                                "total": [],
                                "mse": [],
                                "style": [],
                                "content": [],
                                "style class": [],
                                "discriminate": []
                            }
                        }

        self.dis_history = {
                            "train": [],
                            "valid": []
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

        plt.rcParams["figure.figsize"] = (16, 20)

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

        plt.rcParams["figure.figsize"] = (20, 12)

        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(range(len(self.history["train"]["total"])), self.history["train"]["total"], label="train-loss", color="#00687D")
        ax1.legend()

        ax2 = ax1.twiny()
        ax2.plot(range(len(self.history["valid"]["total"])), self.history["valid"]["total"], label="valid-loss", color="#F07D28")
        ax2.legend()

        plt.savefig(path)


    def train(self):

        avg_loss = {
            "total": 0,
            "mse": 0,
            "style": 0,
            "content": 0,
            "style class": 0,
            "discriminate": 0,
        }

        batch_size = 64
        real_label = torch.ones((batch_size, 1)).to(self.device)
        fake_label = torch.zeros((batch_size, 1)).to(self.device)

        for i in tqdm(range(len(self.trainloader))):
            # ------------------------------------- Discriminator ---------------------------------------
            self.generator.eval()
            self.discriminator.train()

            input_letters, gothic_letters, output_letters, font_labels = self.trainloader.get()

            input_letters = input_letters.to(self.device).type(torch.float32)
            gothic_letters = gothic_letters.to(self.device).type(torch.float32)
            output_letters = output_letters.to(self.device).type(torch.float32)
            font_labels = font_labels.to(self.device)

            fake = self.generator(input_letters, gothic_letters).detach()
            fake_pred, _ = self.discriminator(fake)
            real_pred, real_pred_style = self.discriminator(output_letters)

            d_fake_loss = self.bce_criterion(fake_pred, fake_label)
            d_real_loss = self.bce_criterion(real_pred, real_label)

            d_style_loss = self.cross_entropy(real_pred_style, font_labels)

            d_total_loss = 0.5 * (d_fake_loss + d_real_loss)

            self.dis_optim.zero_grad()
            d_total_loss.backward(retain_graph=True)
            d_style_loss.backward()
            self.dis_optim.step()

            self.dis_history["train"].append(d_total_loss.item())

            # --------------------------------------- Generator -----------------------------------------
            self.generator.train()
            self.discriminator.eval()

            input_letters, gothic_letters, output_letters, font_labels = self.trainloader.get()

            input_letters = input_letters.to(self.device).type(torch.float32)
            gothic_letters = gothic_letters.to(self.device).type(torch.float32)
            output_letters = output_letters.to(self.device).type(torch.float32)
            font_labels = font_labels.to(self.device)

            pred = self.generator(input_letters, gothic_letters)
            discriminator_pred, pred_font_labels = self.discriminator(pred)

            pred_styles = self.generator.style_extractor(pred)
            pred_content = pred
            for i in range(len(self.generator.content_extractor)):
                pred_content = self.generator.content_extractor[i](pred_content)

            mse_loss = F.mse_loss(pred, output_letters)
            style_loss = F.mse_loss(pred_styles, self.generator.style)
            content_loss = F.mse_loss(pred_content, self.generator.content_list[-1])
            style_class_loss = self.cross_entropy(pred_font_labels, font_labels)
            discriminate_loss = self.bce_criterion(discriminator_pred, real_label)

            mse_loss *= self.mse_alpha
            style_loss *= self.style_alpha
            content_loss *= self.content_alpha
            style_class_loss *= self.style_class_alpha
            discriminate_loss *= self.discriminate_alpha

            total_loss = mse_loss + style_loss + content_loss + style_class_loss + discriminate_loss
            
            self.gen_optim.zero_grad()
            total_loss.backward()
            self.gen_optim.step()

            self.history["train"]["total"].append(total_loss.item())
            self.history["train"]["mse"].append(mse_loss.item())
            self.history["train"]["style"].append(style_loss.item())
            self.history["train"]["content"].append(content_loss.item())
            self.history["train"]["style class"].append(style_class_loss.item())
            self.history["train"]["discriminate"].append(discriminate_loss.item())

            avg_loss["total"] += total_loss.item()
            avg_loss["mse"] += mse_loss.item()
            avg_loss["style"] += style_loss.item()
            avg_loss["content"] += content_loss.item()
            avg_loss["style class"] += style_class_loss.item()
            avg_loss["discriminate"] += discriminate_loss.item()

            if i % 50 == 0:
                print(f"mse_loss: {mse_loss.item()}, style_loss: {style_loss.item()}, content_loss: {content_loss.item()}, style_class_loss: {style_class_loss.item()}, discriminate_loss: {discriminate_loss.item()}, d_total_loss: {d_total_loss.item()}")
            
        avg_loss["total"] /= len(self.trainloader)
        avg_loss["mse"] /= len(self.trainloader)
        avg_loss["style"] /= len(self.trainloader)
        avg_loss["content"] /= len(self.trainloader)
        avg_loss["style class"] /= len(self.trainloader)
        avg_loss["discriminate"] /= len(self.trainloader)

        return list(avg_loss.values())

    
    @torch.no_grad()
    def valid(self):
        self.generator.eval()
        self.discriminator.eval()

        avg_loss = {
            "total": 0,
            "mse": 0,
            "style": 0,
            "content": 0,
            "style class": 0,
            "discriminate": 0,
        }

        batch_size = 128
        real_label = torch.ones((batch_size, 1)).to(self.device)
        fake_label = torch.zeros((batch_size, 1)).to(self.device)

        for i in tqdm(range(len(self.validloader))):

            # ------------------------------------- Discriminator ---------------------------------------

            input_letters, gothic_letters, output_letters, font_labels = self.validloader.get()

            input_letters = input_letters.to(self.device).type(torch.float32)
            gothic_letters = gothic_letters.to(self.device).type(torch.float32)
            output_letters = output_letters.to(self.device).type(torch.float32)
            font_labels = font_labels.to(self.device)

            fake = self.generator(input_letters, gothic_letters)
            fake_pred, _ = self.discriminator(fake)
            real_pred, real_pred_style = self.discriminator(output_letters)

            d_fake_loss = self.bce_criterion(fake_pred, fake_label)
            d_real_loss = self.bce_criterion(real_pred, real_label)

            d_style_loss = self.cross_entropy(real_pred_style, font_labels)

            d_total_loss = 0.5 * (d_fake_loss + d_real_loss)

            self.dis_history["valid"].append(d_total_loss.item())

            # --------------------------------------- Generator -----------------------------------------

            input_letters, gothic_letters, output_letters, font_labels = self.validloader.get()

            input_letters = input_letters.to(self.device).type(torch.float32)
            gothic_letters = gothic_letters.to(self.device).type(torch.float32)
            output_letters = output_letters.to(self.device).type(torch.float32)
            font_labels = font_labels.to(self.device)

            pred = self.generator(input_letters, gothic_letters)
            discriminator_pred, pred_font_labels = self.discriminator(pred)

            pred_styles = self.generator.style_extractor(pred)
            pred_content = pred
            for i in range(len(self.generator.content_extractor)):
                pred_content = self.generator.content_extractor[i](pred_content)

            mse_loss = F.mse_loss(pred, output_letters)
            style_loss = F.mse_loss(pred_styles, self.generator.style)
            content_loss = F.mse_loss(pred_content, self.generator.content_list[-1])
            style_class_loss = self.cross_entropy(pred_font_labels, font_labels)
            discriminate_loss = self.bce_criterion(discriminator_pred, real_label)

            mse_loss *= self.mse_alpha
            style_loss *= self.style_alpha
            content_loss *= self.content_alpha
            style_class_loss *= self.style_class_alpha
            discriminate_loss *= self.discriminate_alpha

            total_loss = mse_loss + style_loss + content_loss + style_class_loss + discriminate_loss

            self.history["valid"]["total"].append(total_loss.item())
            self.history["valid"]["mse"].append(mse_loss.item())
            self.history["valid"]["style"].append(style_loss.item())
            self.history["valid"]["content"].append(content_loss.item())
            self.history["valid"]["style class"].append(style_class_loss.item())
            self.history["valid"]["discriminate"].append(discriminate_loss.item())

            avg_loss["total"] += total_loss.item()
            avg_loss["mse"] += mse_loss.item()
            avg_loss["style"] += style_loss.item()
            avg_loss["content"] += content_loss.item()
            avg_loss["style class"] += style_class_loss.item()
            avg_loss["discriminate"] += discriminate_loss.item()
            
        avg_loss["total"] /= len(self.validloader)
        avg_loss["mse"] /= len(self.validloader)
        avg_loss["style"] /= len(self.validloader)
        avg_loss["content"] /= len(self.validloader)
        avg_loss["style class"] /= len(self.validloader)
        avg_loss["discriminate"] /= len(self.validloader)

        if avg_loss["total"] <= self.best_loss:
            self.best_loss = avg_loss["total"]
            self.best_params = self.generator.state_dict()

        return list(avg_loss.values())

    
    def set_alphas(self, mse_alpha, style_alpha, content_alpha, style_class_alpha, discriminate_alpha):
        self.mse_alpha = mse_alpha
        self.style_alpha = style_alpha
        self.content_alpha = content_alpha
        self.style_class_alpha = style_class_alpha
        self.discriminate_alpha = discriminate_alpha


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
            train_total_loss, train_mse_loss, train_style_loss, train_content_loss, train_style_class_loss, train_discriminate_loss = self.train()
            print(f"total_loss: {train_total_loss}, mse_loss: {train_mse_loss}, style_loss: {train_style_loss}, content_loss: {train_content_loss}, style_class_loss: {train_style_class_loss}, discriminate_loss: {train_discriminate_loss}", end="\n\n")

            print("VALID", end="\n")
            valid_total_loss, valid_mse_loss, valid_style_loss, valid_content_loss, valid_style_class_loss, valid_discriminate_loss = self.valid()
            print(f"total_loss: {valid_total_loss}, mse_loss: {valid_mse_loss}, style_loss: {valid_style_loss}, content_loss: {valid_content_loss}, style_class_loss: {valid_style_class_loss}, discriminate_loss: {valid_discriminate_loss}", end="\n\n")
            
            self.save_history(f"{history_base_path}/epoch-{epoch+1}.png")
            self.save_pred(f"{pred_base_path}/epoch-{epoch+1}.png")

            if trainer_base_path != None:
                with open(trainer_base_path + "/trainer.pickle", "wb") as f:
                    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)