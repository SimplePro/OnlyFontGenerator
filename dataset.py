from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split
import torch
import random
from tqdm import tqdm

from random import choice

from os import listdir

import pickle

if __name__ == '__main__':
    from utils import get_random_letter

else:
    from .utils import get_random_letter


def crop_img(image):

    image = np.array(image)
    # 글자의 시작, 끝의 좌표 정의
    text_sx, text_sy = (127, 127)
    text_ex, text_ey = (0, 0)

    # 행을 하나씩 탐색
    for y in range(0, 128):

        # 해당 행의 값이 전부 흰색이 아니라면
        if np.sum(image[y]) != 255*128:

            # 글자의 시작 y좌표는 가장 작은 값임.
            text_sy = min(text_sy, y)

            # 글자의 끝 y좌표는 가장 큰 값임.
            text_ey = max(text_ey, y)

    # 열을 하나씩 탐색
    for x in range(0, 128):

        # 해당 열의 값이 전부 흰색이 아니라면
        if np.sum(image[:, x]) != 255*128:

            # 글자의 시작 x좌표는 가장 작은 값임.
            text_sx = min(text_sx, x)

            # 글자의 끝 x좌표는 가장 큰 값임.
            text_ex = max(text_ex, x)

    # 텍스트 부분만 잘라낸다.
    text_image = image[text_sy:text_ey, text_sx:text_ex]

    # 새로운 이미지를 생성한다.
    croped_image = Image.new(mode="L", size=(128, 128), color=255)
    croped_image = np.array(croped_image)

    # 글자의 가로 길이와 세로 길이를 구한다.
    w = text_ex - text_sx
    h = text_ey - text_sy

    # 글자를 중앙으로 배치하기 위한 좌표를 구한다.
    start_x = (128-w)//2
    end_x = start_x + w

    start_y = (128-h)//2
    end_y = start_y + h

    # 새로운 이미지에 글자를 붙힌다.
    croped_image[start_y:end_y, start_x:end_x] = text_image
    croped_image = Image.fromarray(croped_image)

    return croped_image



# 폰트를 attribute로 하여 글자를 생성하는 클래스.
class Font:
    
    def __init__(self, font_path):
        self.font = ImageFont.truetype(font_path, 100)


    # 글자가 파라메터로 입력되면 이미지로 변환하여 반환한다.
    def text2img(self, letter, tensor=False):
        image = Image.new(mode="L", size=(128, 128), color=255)
        drawing = ImageDraw.Draw(image)

        # 해당 글자를 폰트에서 지원하고 있다면
        try:
            drawing.text(xy=(0, 0), text=letter, fill=(0), font=self.font)
        
        # 아니라면
        except:
            return None
        
        # 이미지를 중앙으로 재배치
        image = crop_img(image)

        if tensor:
            # 텐서로 변환
            image = transforms.ToTensor()(image)

        return image


# Font 데이터셋 클래스
class FontDataset:

    def __init__(
        self,
        style_font_label, # 손글씨 폰트 라벨 (int)
        style_font, # 손글씨 폰트 객체 (Font)
        content_font, # 베이스로할 폰트 객체 (Font)
        length, # 손글씨 데이터셋의 개수 (int)
    ):
        self.style_font_label = style_font_label
        self.style_letters = torch.zeros((length, 1, 128, 128)).type(torch.float16)
        self.content_letters = torch.zeros((length, 1, 128, 128)).type(torch.float16)

        for i in range(length):
            random_letter = get_random_letter()
            self.style_letters[i] = style_font.text2img(random_letter, tensor=True)
            self.content_letters[i] = content_font.text2img(random_letter, tensor=True)


    def __len__(self):
        return len(self.style_letters)



class FontDataLoader:

    def __init__(
        self,
        font_dataset_list, 
        batch_size
    ):
        self.font_dataset_list = font_dataset_list
        self.batch_size = batch_size

    
    def __len__(self):
        return len(self.font_dataset_list) * len(self.font_dataset_list[0]) // self.batch_size


    def get(self, batch_size=None):
        batch_size = self.batch_size if batch_size == None else batch_size

        content_letters = torch.zeros((batch_size, 1, 128, 128)).type(torch.float16)
        style_letters = torch.zeros((batch_size, 1, 128, 128)).type(torch.float16)
        style_labels = torch.zeros((batch_size, 1)).type(torch.float16)

        for i in range(batch_size):

            random_font_dataset = choice(self.font_dataset_list)

            random_sample_idx = choice(range(len(random_font_dataset)))

            content_letters[i] = random_font_dataset.content_letters[random_sample_idx]
            style_letters[i] = random_font_dataset.style_letters[random_sample_idx]
            style_labels[i] = random_font_dataset.style_font_label

        return (content_letters, style_letters, style_labels)



if __name__ == '__main__':

    nanum_font = Font("./data/font-ttf/NanumGothic.ttf")

    # base_path = "data/font-ttf/"
    # font_path_list = [f"{base_path}{font_file}" for font_file in listdir(base_path)]
    # print(len(font_path_list))

    # train_dataset_list = [
    #     FontDataset(
    #         style_font_label=i,
    #         style_font=Font(font_path),
    #         content_font=nanum_font,
    #         length=3000
    #     )
    #     for (i, font_path) in tqdm(enumerate(font_path_list))
    # ]

    # valid_dataset_list = [
    #     FontDataset(
    #         style_font_label=i,
    #         style_font=Font(font_path),
    #         content_font=nanum_font,
    #         length=300
    #     )
    #     for (i, font_path) in tqdm(enumerate(font_path_list))
    # ]

    # trainloader = FontDataLoader(train_dataset_list, batch_size=1)

    # for i in range(10):
    #     content_letters, style_letters, style_labels = trainloader.get()

    #     print(font_path_list[int(style_labels.item())])
    #     transforms.ToPILImage()(content_letters[0]).show()
    #     transforms.ToPILImage()(style_letters[0]).show()

    # if int(input("If you don't want to save dataset, enter number 0: ")) != 0:
    # with open("data/train_dataset_list.pickle", "wb") as f:
    #     pickle.dump(train_dataset_list, f, pickle.HIGHEST_PROTOCOL)

    # with open("data/valid_dataset_list.pickle", "wb") as f:
    #     pickle.dump(valid_dataset_list, f, pickle.HIGHEST_PROTOCOL)


    ga_style_letters_img = Image.open("./ga_style_letters.png").convert("L").resize((128, 128))
    
    ga_style_letters_img = crop_img(ga_style_letters_img)
    ga_style_letters_img.save("./ga_style_letters.png")

    gam_style_letters_img = Image.open("./gam_style_letters.png").convert("L").resize((128, 128))
    
    gam_style_letters_img = crop_img(gam_style_letters_img)
    gam_style_letters_img.save("./gam_style_letters.png")

    him_style_letters_img = Image.open("./him_style_letters.png").convert("L").resize((128, 128))
    
    him_style_letters_img = crop_img(him_style_letters_img)
    him_style_letters_img.save("./him_style_letters.png")

    la_style_letters_img = Image.open("./la_style_letters.png").convert("L").resize((128, 128))
    
    la_style_letters_img = crop_img(la_style_letters_img)
    la_style_letters_img.save("./la_style_letters.png")

    nanum_font.text2img("가").save("./ga_content_letters.png")
    nanum_font.text2img("감").save("./gam_content_letters.png")
    nanum_font.text2img("힘").save("./him_content_letters.png")
    nanum_font.text2img("라").save("./la_content_letters.png")