from vocabulary import Vocabulary
from beheaded_inception3 import beheaded_inception_v3
from model import CaptionDecoderRNN
import torch

vocab = Vocabulary(vocab_threshold=5, vocab_from_file=True)
inception = beheaded_inception_v3().train(False)
model = CaptionDecoderRNN(embed_size=300,
                              hidden_size=512,
                              vocab_size=len(vocab),
                              cnn_feature_size=2048,
                              num_layers=4)
model.load_state_dict(torch.load('./best_model_base.ckpt', map_location=torch.device('cpu')))

model.eval()