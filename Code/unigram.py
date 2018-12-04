import pickle
from math import log10


# TODO: when testing, try to load file from pickle, train if file does not exist.
class UnigramModel:
    SMOOTHING_DEFAULT = 0.5

    def __init__(self, input_str=None, smoothing=0.5):
        self.char_dict = {}
        self.probs_dict = {}
        self.training_size = 0
        self.trained = False
        self.smoothing = smoothing
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

            self.trained = True

    def _calc_probs(self):

        if self.trained:
            for char in self.char_dict:
                self.probs_dict[char] = self.calc_prob(char_count=self.char_dict[char],
                                                       total_count=self.training_size,
                                                       smoothing=self.smoothing,
                                                       vocab_size=len(self.char_dict))

    # Parse the input for any unaccounted characters for smoothing
    def _calc_probs_with_smoothing(self, input_str):
        if self.trained:

            for char in input_str:
                if char not in self.char_dict:
                    self.char_dict[char] = 0

            self._calc_probs()

    def prob_sentence(self, input_str):

        total_prob = 0
        result_cumul = []
        result_single = {}
        if self.trained:

            self._calc_probs_with_smoothing(input_str)

            for char in input_str:
                if char in self.char_dict:
                    current_prob = self.probs_dict[char]
                    result_single[char] = current_prob
                    total_prob += current_prob
                    result_cumul.append((char, total_prob))

        else:
            print('Must train model before attempting to evaluate a sentence!')

        return [total_prob, result_single, result_cumul]

    @staticmethod
    def calc_prob(char_count, total_count, smoothing=0.0, vocab_size=0.0):

        if smoothing < 0.0:
            smoothing = UnigramModel.SMOOTHING_DEFAULT

        try:
            return log10((char_count + smoothing) / (total_count + smoothing * vocab_size))
        except ValueError:
            return 0.0
