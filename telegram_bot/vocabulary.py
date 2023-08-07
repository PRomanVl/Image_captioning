import pickle
import os


class Vocabulary(object):
    def __init__(self,
                 vocab_threshold,
                 vocab_file='vocab.pkl',
                 start_word='<bos>',
                 end_word='<eos>',
                 pad_word='<pad>',
                 unk_word='<unk>',
                 path_tokenized_file='captions_tokenized.json',
                 vocab_from_file=False):

        ''' Инициализация парамметров словаря.

            Параметры:
            vocab_threshold: Минимальное количество слов для добавления в словарь
                             слова.
            vocab_file: Файл со словарем.
            start_word: Специальный символ для указания начала в предложении.
            end_word: Специальный символ для указания конца в предложении.
            pad_word: Специальный символ для указания пропуска в предложении.
            unk_word: Специальный символ для указания неизвестного слова
                      в предложении.
            capt_tokenized_file: Путь до токенизированных данных.
            vocab_from_file: Если False, то создает словарь и записывает его
                             в vocab_file. Если True, то подгружает его из
                             vocab_file если он существует.
        '''
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.pad_word = pad_word
        self.unk_word = unk_word
        self.path_tokenized_file = path_tokenized_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        '''Загружает файл словаря или создает его и сохраняет'''
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab_file = pickle.load(f)
                self.word2idx = vocab_file.word2idx
                self.idx2word = vocab_file.idx2word
                print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        '''Создание словаря и наполнение его словами'''
        self.init_vocab()
        self.add_word(self.unk_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.pad_word)
        self.add_captions()

    def init_vocab(self):
        '''Инициализация словаря'''
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        '''Добавление токена и количества в словарь'''
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        '''Цикл для добавления слов из всего текста в словарь с ограничением по
           количеству слов'''
        captions = json.load(open(self.path_tokenized_file))
        counter = Counter()

        for img_i in range(len(captions)):
            for caption_i in range(len(captions[img_i])):
                sentence = captions[img_i][caption_i]
                counter.update(sentence.split(' '))

            if img_i % 50000 == 0:
                print("[%d/%d] Building vocab..." % (img_i, len(captions)))

        print('[%d/%d] Finish!' % (len(captions), len(captions)))

        words = [word for word, count in counter.items()
                 if count >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def get_idx(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def get_word(self, idx):
        if not idx in self.idx2word:
            return self.unk_word
        return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)

