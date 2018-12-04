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

        total_count = 0
        for char_count in unigram_model.char_dict.values():
            total_count += char_count

        self.assertEqual(total_count, len(UnigramTests.train_str))

        self.assertTrue(unigram_model.trained)

    def test_prob(self):

        train_str_unique = UnigramTests.str_unique_chars(UnigramTests.train_str)
        unigram_model = UnigramModel(UnigramTests.train_str, smoothing=0.0)
        self.assertEqual(len(unigram_model.char_dict), len(train_str_unique))

        unigram_model._calc_probs()

        # test one of the probability calculations
        # For character 'z'
        prob_test = log10(1.0 / len(UnigramTests.train_str))
        self.assertAlmostEqual(unigram_model.probs_dict['z'], prob_test, places=2)

    def test_unigram_train_smoothed(self):
        unigram_model = UnigramModel(UnigramTests.train_str)

        unigram_model._calc_probs()

        # test one of the probability calculations
        # For character 'z'
        prob_test = log10((1.0 + UnigramModel.SMOOTHING_DEFAULT) /
                          (len(UnigramTests.train_str) + UnigramModel.SMOOTHING_DEFAULT * len(unigram_model.char_dict)))
        self.assertAlmostEqual(unigram_model.probs_dict['z'], prob_test, places=2)

    def test_unigram_test_str(self):
        unigram_model = UnigramModel(UnigramTests.train_str, smoothing=0.0)
        unigram_model._calc_probs()
        self.assertEqual(len(unigram_model.probs_dict), len(unigram_model.char_dict))

        expected_val = 0.0
        for char in UnigramTests.input_str:
            if char in unigram_model.probs_dict:
                expected_val += unigram_model.probs_dict[char]

        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str)[0], expected_val, places=2)

    def test_unigram_test_str_smooth(self):
        unigram_model = UnigramModel(UnigramTests.train_str)

        expected_val = 0.0
        char_dict = dict()
        char_dict.update(unigram_model.char_dict)

        for char in UnigramTests.input_str:
            char_dict[char] = 0

        for char in UnigramTests.input_str:
            if char in unigram_model.char_dict:
                expected_val += UnigramModel.calc_prob(char_count=unigram_model.char_dict[char],
                                                       total_count=unigram_model.training_size,
                                                       smoothing=UnigramModel.SMOOTHING_DEFAULT,
                                                       vocab_size=len(char_dict))
            else:
                expected_val += UnigramModel.calc_prob(char_count=0,
                                                       total_count=unigram_model.training_size,
                                                       smoothing=UnigramModel.SMOOTHING_DEFAULT,
                                                       vocab_size=len(char_dict))

        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str)[0], expected_val, places=2)

    def test_unigram_test_str_untrained(self):
        unigram_model = UnigramModel(smoothing=0.0)
        unigram_model_smooth = UnigramModel()

        unigram_model._calc_probs()
        unigram_model_smooth._calc_probs()

        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str)[0], 0.0, places=2)
        self.assertAlmostEqual(unigram_model_smooth.prob_sentence(UnigramTests.input_str)[0], 0.0, places=2)

    # Useful functions
    @staticmethod
    def str_unique_chars(input_str):
        return ''.join(set(input_str))


if __name__ == '__main__':
    unittest.main()
