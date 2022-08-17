from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split
import torch
import random
from tqdm import tqdm

from random import randint
from jamo import h2j, j2hcj
from unicode import join_jamos

from os import listdir

import pickle


FIRST_LETTER = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
MIDDLE_LETTER = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ','ㅢ', 'ㅣ']
LAST_LETTER = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

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
class FontImage:
    
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


default_font = FontImage("./data/font-ttf/NanumGothic.ttf")


def get_random_letter():
    first_letter = FIRST_LETTER[randint(0, len(FIRST_LETTER)-1)]
    middle_letter = MIDDLE_LETTER[randint(0, len(MIDDLE_LETTER)-1)]
    last_letter = LAST_LETTER[randint(0, len(LAST_LETTER)-1)]

    return join_jamos(first_letter + middle_letter + last_letter)


class FontDataset:

    def __init__(self, font, font_path, size):
        self.font_path = font_path
        self.letters = torch.zeros((size, 1, 128, 128)).type(torch.float16)
        self.gothic_letters = torch.zeros((size, 1, 128, 128)).type(torch.float16)

        for i in range(size):
            random_letter = get_random_letter()
            self.letters[i] = font.text2img(random_letter, tensor=True)
            self.gothic_letters[i] = default_font.text2img(random_letter, tensor=True)



class FontLoader:

    def __init__(self, font_dataset_list, batch_size):
        self.font_dataset_list = font_dataset_list
        self.batch_size = batch_size

    
    def __len__(self):
        return len(self.font_dataset_list) * len(self.font_dataset_list[0].letters) // self.batch_size


    def get(self, batch_size=None):
        if batch_size == None: batch_size = self.batch_size

        input_letters, gothic_letters, output_letters = torch.zeros((3, batch_size, 1, 128, 128)).type(torch.float16)
        font_labels = torch.zeros((batch_size)).type(torch.long)

        for i in range(batch_size):
            
            font_idx = randint(0, len(self.font_dataset_list)-1)
            random_font = self.font_dataset_list[font_idx]
            font_labels[i] = font_idx

            input_letters[i] = random_font.letters[randint(0, len(random_font.letters)-1)]

            output_idx = randint(0, len(random_font.letters)-1)
            gothic_letters[i] = random_font.gothic_letters[output_idx]
            output_letters[i] = random_font.letters[output_idx]

        return (input_letters, gothic_letters, output_letters, font_labels)


if __name__ == '__main__':

    # base_path = "data/font-ttf/"
    # font_file_list = listdir(base_path)

    # train_dataset_list = []
    # valid_dataset_list = []

    # for font_file in tqdm(font_file_list):
    #     font_path = f"{base_path}{font_file}"
    #     font = FontImage(font_path)
    #     train_dataset_list.append(FontDataset(font, font_path, size=500))
    #     valid_dataset_list.append(FontDataset(font, font_path, size=50))

    # with open("data/train_dataset_list.pickle", "wb") as f:
    #     pickle.dump(train_dataset_list, f, pickle.HIGHEST_PROTOCOL)

    # with open("data/valid_dataset_list.pickle", "wb") as f:
    #     pickle.dump(valid_dataset_list, f, pickle.HIGHEST_PROTOCOL)

    # han_donghwan_img = Image.open("./han_donghwan.png").convert("L").resize((128, 128))
    
    # han_donghwan_img = crop_img(han_donghwan_img)
    # han_donghwan_img.save("./han_donghwan.png")

    default_font.text2img("한").save("./han_gothic.png")
    default_font.text2img("글").save("./guel_gothic.png")