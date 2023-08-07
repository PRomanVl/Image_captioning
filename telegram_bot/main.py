from app import generate_caption
from vocabulary import Vocabulary
from beheaded_inception3 import beheaded_inception_v3
from model import CaptionDecoderRNN
import torch

vocab = Vocabulary(vocab_threshold=5, vocab_from_file=True)
image_decoder = beheaded_inception_v3().train(False)
model = CaptionDecoderRNN(embed_size=300,
                              hidden_size=512,
                              vocab_size=len(vocab),
                              cnn_feature_size=2048,
                              num_layers=4)

model.load_state_dict(torch.load('C:/best_model_base.ckpt', map_location=torch.device('cpu')))

model.eval()

import os

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
# bot = Bot(token=os.getenv('TOKEN'))
bot = Bot('5042981708:AAEhFh05by4ZUtx9Vlxm4IyIv2S4uARYGm8')
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет!\nЯ могу угадать что изображено на картинке \n Пришли мне фотографию, и скажу что на ней")


@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(msg):
    # download user img
    download_dir = './img/'
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    img_name = 'img'+str(msg.from_user.id)+'.jpg'
    print(img_name)
    await msg.photo[-1].download('./img/' + img_name)
    # predict

    for i in range(1):
        answer = ' '.join(generate_caption('./img/' + img_name, vocab, model, image_decoder, t=5, device='cpu')[1:-1])
    print(answer)
    # delete user img
    os.remove('./img/' + img_name)
    await bot.send_message(chat_id=msg.from_user.id, text=answer)


executor.start_polling  (dp, skip_updates=True)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    executor.start_polling(dp)