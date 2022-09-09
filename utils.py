import pickle

from random import randint

with open("utils/AllLetters.pickle", "rb") as f:
    all_letters = pickle.load(f)


def get_random_letter():
    random_idx = randint(0, len(all_letters)-1)
    return all_letters[random_idx]



def make_all_letters_pkl():
    all_letters = []
    with open("utils/AllLetters.txt", "r", encoding="UTF8") as f:
        all_letters.extend([i for i in f.readline()])
    
    with open("utils/AllLetters.pickle", "wb") as f:
        pickle.dump(all_letters[1:], f)



if __name__ == '__main__':
    make_all_letters_pkl()