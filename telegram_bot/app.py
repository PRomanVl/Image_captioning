import PIL
import torch
import pickle
import os
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from beheaded_inception3 import beheaded_inception_v3
from model import CaptionDecoderRNN
# from vocabulary import Vocabulary


def transform(image, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_image = image

    base_transform = T.Compose([T.ToTensor(),
                                T.Resize(img_size),
                                T.Normalize(mean=mean, std=std)])
    new_image = base_transform(new_image)

    return new_image



def generate_caption(input_img_path, vocab, model, image_decoder, t=5, max_len=100, device='cpu'):
    print(input_img_path)


    image = PIL.Image.open(input_img_path)
    image = transform(image, [299, 299])
    print(type(image))
    print(image.size())
    # image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

    vectors_8x8, vectors_neck, logits = image_decoder(image[None])
    print('image coding is ok')
    captions_prefix = ['<bos>']

    for _ in range(max_len):
        # Получаем коды слов из словаря
        captions_ix = [vocab.get_idx(word) for word in captions_prefix]
        # Оборачиваем в торч тензор
        captions_ix = torch.tensor(captions_ix, dtype=torch.int64, device=device)

        # Получаем логиты следующего слова
        next_word_logits = model.forward(vectors_neck.to(device), captions_ix.unsqueeze(0))[0, -1]
        # Преобразуем их в вероятности
        next_word_probs = F.softmax(next_word_logits.cpu(), -1).data.numpy()

        assert len(next_word_probs.shape) == 1

        # Не совсем понял что это, но в домашкe это было.
        # Видимо очень важно и нужно!
        next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t)

        # Получаем слово в исходном виде
        next_word = vocab.get_word(np.argmax(next_word_probs).item())

        # Добавляем его к предложению
        captions_prefix.append(next_word)

        # Если сгенерировалось конец предложения, то останавливаем процесс
        if next_word == '<eos>':
            break

    return captions_prefix


