import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.inception import Inception3
from warnings import warn


class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else:
            warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x


from torch.utils.model_zoo import load_url


def beheaded_inception_v3(transform_input=True):
    model = BeheadedInception3(transform_input=transform_input)
    inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    model.load_state_dict(load_url(inception_url))
    return model


class CaptionDecoderRNN(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 cnn_feature_size=2048,
                 num_layers=2):
        super(self.__class__, self).__init__()

        ''' Декодер для описания изображения (вектора представления)

            Параметры:
            embed_size: Размер эмбединга
            hidden_size: Размер скрытого слоя
            vocab_size: Размер словаря
            cnn_feature_size: Размер выходного вектора
            num_layers: Количество слоев
        '''

        # стандартная архитектура такой сети такая:
        # 1. линейные слои для преобразования эмбеддиинга картинки в начальные состояния h0 и c0 LSTM-ки
        # 2. слой эмбедднга
        # 3. несколько LSTM слоев (для начала не берите больше двух, чтобы долго не ждать)
        # 4. линейный слой для получения логитов
        self.num_layers = num_layers

        self.img_to_h0 = nn.Linear(cnn_feature_size, hidden_size)
        self.img_to_h0 = nn.Linear(cnn_feature_size, hidden_size)
        self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.2)

        self.emb = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.logits = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_vectors, captions):
        """
        Apply the network in training mode.
        :param image_vectors: torch tensor, содержащий выходы inseption.
               shape: [batch_size, cnn_feature_size]

        :param captions_idx: таргет описания картинок в виде матрицы
               shape: [batch_size, word_i]

        :returns: outputs: логиты для сгенерированного текста описания
                  shape: [batch_size, word_i, p_vocab_i]
        """
        # 1. инициализируем LSTM state
        # [batch_size, hidden_size]
        h0 = self.tanh(self.img_to_h0(image_vectors))
        c0 = self.tanh(self.img_to_h0(image_vectors))

        # make [num_layer, batch_size, hidden_size]
        h0 = h0.expand(self.num_layers, h0.size(0), h0.size(1)).contiguous()
        c0 = c0.expand(self.num_layers, c0.size(0), c0.size(1)).contiguous()

        # 2. применим слой эмбеддингов к captions
        # [batch_size, max_len, emb_size]
        captions_emb = self.emb(captions)

        # 3. кормим LSTM captions_emb и добавим скрытые веса картинки
        # [batch_size, max_len, hidden_size]
        hiddens, _ = self.lstm(captions_emb, (h0, c0))

        # 4. посчитаем логиты из выхода LSTM
        # [batch_size, max_len, vocab_size]
        logits = self.logits(hiddens)

        return logits

