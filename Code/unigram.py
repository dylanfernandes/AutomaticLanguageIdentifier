import pickle
from math import log10


# TODO: when testing, try to load file from pickle, train if file does not exist.
class UnigramModel:
    SMOOTHING = 0.5

    def __init__(self, input_str=None, smooth=False, smoothing=-1):
        self.char_dict = {}
        self.probs = {}
        self.training_size = 0
        self.trained = False
        self.smooth = smooth
        if input_str:
            self.train(input_str, smoothing)

    def train(self, input_str, smoothing=-1):
        # TODO: save to pickle

        if input_str:
            self.training_size = len(input_str)
            for char in input_str:
                if char in self.char_dict:
                    self.char_dict[char] += 1
                else:
                    self.char_dict[char] = 1

            self._calc_probs(smoothing)

            self.trained = True

    def _calc_probs(self, smoothing=-1):

        for char in self.char_dict:
            if self.smooth:
                self.probs[char] = self.calc_prob(char_count=self.char_dict[char],
                                                  total_count=self.training_size,
                                                  smoothing=smoothing,
                                                  vocab_size=len(self.char_dict))
            else:
                self.probs[char] = self.calc_prob(char_count=self.char_dict[char], total_count=self.training_size)

    # Parse the input for any unaccounted characters for smoothing
    def _parse_input_smoothing(self, input_str, smoothing=-1):
        if self.trained:

            original_vocab_size = len(self.char_dict)

            for char in input_str:
                if char not in self.char_dict:
                    self.char_dict[char] = 0

            if original_vocab_size != len(self.char_dict):
                self._calc_probs(smoothing)

    def prob_sentence(self, input_str, smoothing=-1):

        total_prob = 0
        if self.trained:

            if self.smooth:
                self._parse_input_smoothing(input_str, smoothing)

            for char in input_str:
                if char in self.char_dict:
                    total_prob += self.probs[char]
        else:
            print('Model has not been trained!')

        return total_prob

    @staticmethod
    def calc_prob(char_count, total_count, smoothing=0.0, vocab_size=0.0):

        if smoothing < 0.0:
            smoothing = UnigramModel.SMOOTHING

        return log10((char_count + smoothing) / (total_count + smoothing * vocab_size))
