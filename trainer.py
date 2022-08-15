import pickle

from model import Generator, Loss
from torch.nn.optim import Adam


class Trainer:

    def __init__(
                self,
                model, # 모델
                device, # 디바이스
                trainloader, # 학습 로더
                validloader, # 검증 로더
                lr=0.002, # 학습률
                vgg_alpha=1,
                mse_alpha=1,
            ):

        self.model = model.to(device)
        self.criterion = Loss(device=device)
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas)

        self.trainloader = trainloader
        self.validloader = validloader

        self.history = {
                            "train": torch.zeros((3, 1)), # (total_loss, vgg_loss, mse_loss)
                            "valid": torch.zeros((3, 1)) # (total_loss, vgg_loss, mse_loss)
                        }
        
        

        self.best_loss = 1e5
        self.best_params = None

        self.log_face_style = self.validloader.faceset[:3][0].to(device).type(torch.float32)
        self.log_expression_landmarks = convert_landmarks22d(batch_size=3, landmarks=self.validloader.landmarkset[:3][0]).to(device).type(torch.float32)
    

    @torch.no_grad()
    def save_pred(self, path):
        plt.clf()

        plt.rcParams["figure.figsize"] = (16, 16)

        self.model.eval()

        face_images = (self.log_face_style.cpu().detach().numpy() * 255).astype(np.uint)
        expression_images = (self.log_expression_landmarks.cpu().detach().numpy() * 255).astype(np.uint)
        pred_images = (self.model(self.log_face_style, self.log_expression_landmarks).cpu().detach().numpy() * 255).astype(np.uint)

        for i in range(3):
            face_img = face_images[i].transpose(1, 2, 0)
            expression_img = expression_images[i].transpose(1, 2, 0).reshape(128, 128)
            pred_img = pred_images[i].transpose(1, 2, 0)

            plt.subplot(3, 3, 3*i+1)
            plt.imshow(face_img)
            plt.axis("off")

            plt.subplot(3, 3, 3*i+2)
            plt.imshow(expression_img)
            plt.axis("off")

            plt.subplot(3, 3, 3*i+3)
            plt.imshow(pred_img)
            plt.axis("off")

        plt.savefig(path, dpi=300)


    @torch.no_grad()
    def save_history(self, path):

        plt.clf()

        train_size = self.history["train"][0].size(0)

        plt.rcParams["figure.figsize"] = (20, 12)

        ax1 = plt.subplot(1, 1, 1)

        ax1.plot(range(train_size-1), self.history["train"][0][1:], label="train-loss", color="#00687D")
        ax1.legend()

        valid_size = self.history["valid"][0].size(0)

        ax2 = ax1.twiny()

        ax2.plot(range(valid_size-1), self.history["valid"][0][1:], label="valid-loss", color="#F07D28")
        
        ax2.legend()
        plt.savefig(path)


    def train(self, epoch):
        self.model.train()

        avg_loss = torch.zeros((4, 1))

        for i in tqdm(range(len(self.trainloader))):

            face_style, face_landmarks, (expression_1d_landmarks, expression_2d_landmarks) = self.trainloader.get()

            face_style = face_style.to(device).type(torch.float32)
            face_landmarks = face_landmarks.to(device).type(torch.float32)
            expression_1d_landmarks = expression_1d_landmarks.to(device).type(torch.float32)
            expression_2d_landmarks = expression_2d_landmarks.to(device).type(torch.float32)

            pred = self.model(face_style, expression_2d_landmarks)

            face_loss, landmark_loss, background_loss = self.criterion(pred, face_style, face_landmarks, expression_1d_landmarks)
            face_loss *= self.face_alpha
            landmark_loss *= self.landmark_alpha
            background_loss *= self.background_alpha
            
            self.optim.zero_grad()

            if i % 2 == 0:
                for p in self.model.style_extractor.parameters():
                    p.requires_grad = False

                for p in self.model.expression_extractor.parameters():
                    p.requires_grad = True

                landmark_loss.backward()

            elif i % 2 == 1:
                for p in self.model.style_extractor.parameters():
                    p.requires_grad = True

                for p in self.model.expression_extractor.parameters():
                    p.requires_grad = False

                face_loss.backward(retain_graph=True)
                background_loss.backward()

            self.optim.step()

            total_loss = face_loss + landmark_loss + background_loss
            loss_tensor = torch.Tensor([total_loss.item(), face_loss.item(), landmark_loss.item(), background_loss.item()]).reshape(4, 1)

            avg_loss += loss_tensor

        avg_loss /= len(self.trainloader)

        self.history["train"] = torch.cat((self.history["train"], avg_loss), dim=1)

        return [i.item() for i in avg_loss]

    
    @torch.no_grad()
    def valid(self):
        self.model.eval()

        avg_loss = torch.zeros((4, 1))

        for i in tqdm(range(len(self.validloader))):
            face_style, face_landmarks, (expression_1d_landmarks, expression_2d_landmarks) = self.validloader.get()

            face_style = face_style.to(device).type(torch.float32)
            face_landmarks = face_landmarks.to(device).type(torch.float32)
            expression_1d_landmarks = expression_1d_landmarks.to(device).type(torch.float32)
            expression_2d_landmarks = expression_2d_landmarks.to(device).type(torch.float32)

            pred = self.model(face_style, expression_2d_landmarks)

            face_loss, landmark_loss, background_loss = self.criterion(pred, face_style, face_landmarks, expression_1d_landmarks)

            face_loss *= self.face_alpha
            landmark_loss *= self.landmark_alpha
            background_loss *= self.background_alpha

            total_loss = face_loss + landmark_loss + background_loss

            loss_tensor = torch.Tensor([total_loss.item(), face_loss.item(), landmark_loss.item(), background_loss.item()]).reshape(4, 1)

            avg_loss += loss_tensor

        avg_loss /= len(self.validloader)

        self.history["valid"] = torch.cat((self.history["valid"], avg_loss), dim=1)

        if avg_loss[0] < self.best_loss:
            self.best_loss = avg_loss[0]
            self.best_params = self.model.state_dict()

        return [i.item() for i in avg_loss]


    def set_loss_alpha(self, face_alpha, landmark_alpha, background_alpha):
        self.face_alpha = face_alpha
        self.landmark_alpha = landmark_alpha
        self.background_alpha = background_alpha


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
            train_avg_loss, train_avg_face_loss, train_avg_landmark_loss, train_avg_background_loss = self.train(epoch)
            print(f"loss: {train_avg_loss}, face_loss: {train_avg_face_loss}, landmark_loss: {train_avg_landmark_loss}, background_loss: {train_avg_background_loss}", end="\n\n")

            print("VALID", end="\n")
            valid_avg_loss, valid_avg_face_loss, valid_avg_landmark_loss, valid_avg_background_loss = self.valid()
            print(f"loss: {valid_avg_loss}, face_loss: {valid_avg_face_loss}, landmark_loss: {valid_avg_landmark_loss}, background_loss: {valid_avg_background_loss}", end="\n\n\n")

            self.save_history(f"{history_base_path}/epoch-{epoch+1}.png")
            self.save_pred(f"{pred_base_path}/epoch-{epoch+1}.png")

            if trainer_base_path != None:
                with open(trainer_base_path + "/trainer.pickle", "wb") as f:
                    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)