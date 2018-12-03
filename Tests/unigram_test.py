import unittest
import sys
sys.path.append('../')
from Code.unigram import UnigramModel
from math import log10


class UnigramTests(unittest.TestCase):

    train_str = "zahhjewjkojfcfdssd"
    input_str = "aaabbcdddd"

    def test_train(self):

        train_str_unique = UnigramTests.str_unique_chars(UnigramTests.train_str)
        unigram_model = UnigramModel(UnigramTests.train_str)
        self.assertEqual(len(unigram_model.char_dict), len(train_str_unique))
        self.assertEqual(len(unigram_model.probs), len(unigram_model.char_dict))

        total_count = 0
        for char_count in unigram_model.char_dict.values():
            total_count += char_count

        self.assertEqual(total_count, len(UnigramTests.train_str))

        # test one of the probability calculations
        # For character 'z'
        prob_test = log10(1.0 / len(UnigramTests.train_str))
        self.assertEqual(unigram_model.probs['z'], prob_test)

        self.assertTrue(unigram_model.trained)

    def test_unigram_train_smoothed(self):
        unigram_model = UnigramModel(UnigramTests.train_str, True)

        # test one of the probability calculations
        # For character 'z'
        prob_test = log10((1.0 + UnigramModel.SMOOTHING) /
                          (len(UnigramTests.train_str) + UnigramModel.SMOOTHING * len(unigram_model.char_dict)))
        self.assertEqual(unigram_model.probs['z'], prob_test)

    def test_unigram_test_str(self):
        unigram_model = UnigramModel(UnigramTests.train_str)

        expected_val = 0.0
        for char in UnigramTests.input_str:
            if char in unigram_model.probs:
                expected_val += unigram_model.probs[char]

        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str), expected_val, places=2)

    def test_unigram_test_str_smooth(self):
        unigram_model = UnigramModel(UnigramTests.train_str, True)

        expected_val = 0.0
        char_dict = dict()
        char_dict.update(unigram_model.char_dict)

        for char in UnigramTests.input_str:
            char_dict[char] = 0

        for char in UnigramTests.input_str:
            if char in unigram_model.char_dict:
                expected_val += UnigramModel.calc_prob(char_count=unigram_model.char_dict[char],
                                                       total_count=unigram_model.training_size,
                                                       smoothing=UnigramModel.SMOOTHING,
                                                       vocab_size=len(char_dict))
            else:
                expected_val += UnigramModel.calc_prob(char_count=0,
                                                       total_count=unigram_model.training_size,
                                                       smoothing=UnigramModel.SMOOTHING,
                                                       vocab_size=len(char_dict))

        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str), expected_val, places=2)

    def test_unigram_test_str_untrained(self):
        unigram_model = UnigramModel()
        unigram_model_smooth = UnigramModel(smooth=True)

        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str), 0.0, places=2)
        self.assertAlmostEqual(unigram_model_smooth.prob_sentence(UnigramTests.input_str), 0.0, places=2)

    # Useful functions
    @staticmethod
    def str_unique_chars(input_str):
        return ''.join(set(input_str))


if __name__ == '__main__':
    unittest.main()
