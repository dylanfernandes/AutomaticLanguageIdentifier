import pickle
from math import log10


# TODO: when testing, try to load file from pickle, train if file does not exist.
class UnigramModel:
    SMOOTHING = 0.5
    VOCAB_SIZE = 26

    def __init__(self, input_str=None, smooth=True):
        self.char_dict = {}
        self.probs = {}
        self.training_size = 0
        self.trained = False
        self.smooth = smooth
        if input_str:
            self.train(input_str)

    def train(self, input_str):
        # TODO: save to pickle

        if input_str:
            self.training_size = len(input_str)
            for char in input_str:
                if char in self.char_dict:
                    self.char_dict[char] += 1
                else:
                    self.char_dict[char] = 1

            for char in self.char_dict:
                if self.smooth:
                    self.probs[char] = self.calc_prob(char_count=self.char_dict[char],
                                                      total_count=self.training_size,
                                                      smoothing=UnigramModel.SMOOTHING,
                                                      vocab_size=UnigramModel.VOCAB_SIZE)
                else:
                    self.probs[char] = self.calc_prob(char_count=self.char_dict[char], total_count=self.training_size)

            self.trained = True

    def prob_sentence(self, input_str):

        total_prob = 0
        if self.trained:
            for char in input_str:
                if char in self.char_dict:
                    total_prob += self.probs[char]
                elif self.smooth:
                    total_prob += self.calc_prob(0, total_count=self.training_size, smoothing=UnigramModel.SMOOTHING,
                                                 vocab_size=UnigramModel.VOCAB_SIZE)
        else:
            print('Model has not been trained!')

        return total_prob

    @staticmethod
    def calc_prob(char_count, total_count, smoothing=0.0, vocab_size=0.0):

        if smoothing < 0.0:
            smoothing = UnigramModel.SMOOTHING

        if vocab_size < 0.0:
            vocab_size = UnigramModel.VOCAB_SIZE

        return log10((char_count + smoothing) / (total_count + smoothing * vocab_size))
