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

        test_sentence_op = unigram_model.get_string_prob(UnigramTests.input_str)
        self.assertEqual(len(unigram_model.probs_dict), len(unigram_model.char_dict))

        total_prob_exp = 0.0
        cumul_prob_exp = []
        single_prob_exp = {}
        for char in UnigramTests.input_str:
            if char in unigram_model.probs_dict:
                current_prob = UnigramModel.calc_prob(char_count=unigram_model.char_dict[char],
                                                      total_count=unigram_model.training_size)
                total_prob_exp += current_prob
                cumul_prob_exp.append((char, total_prob_exp))
                single_prob_exp[char] = current_prob

        self.assertAlmostEqual(test_sentence_op[0], total_prob_exp, places=2)
        self.assertEqual(single_prob_exp, test_sentence_op[1])
        self.assertEqual(cumul_prob_exp, test_sentence_op[2])

    def test_unigram_test_str_smooth(self):
        unigram_model = UnigramModel(UnigramTests.train_str)

        test_sentence_op = unigram_model.get_string_prob(UnigramTests.input_str)

        total_prob_exp = 0.0
        cumul_prob_exp = []
        single_prob_exp = {}
        char_dict = dict()
        char_dict.update(unigram_model.char_dict)

        for char in UnigramTests.input_str:
            char_dict[char] = 0

        for char in UnigramTests.input_str:
            if char in unigram_model.char_dict:
                current_prob = UnigramModel.calc_prob(char_count=unigram_model.char_dict[char],
                                                      total_count=unigram_model.training_size,
                                                      smoothing=UnigramModel.SMOOTHING_DEFAULT,
                                                      vocab_size=len(char_dict))
            else:
                current_prob = UnigramModel.calc_prob(char_count=0,
                                                      total_count=unigram_model.training_size,
                                                      smoothing=UnigramModel.SMOOTHING_DEFAULT,
                                                      vocab_size=len(char_dict))
            single_prob_exp[char] = current_prob
            total_prob_exp += current_prob
            cumul_prob_exp.append((char, total_prob_exp))

        self.assertEqual(len(char_dict), len(unigram_model.char_dict))
        self.assertEqual(len(unigram_model.probs_dict), len(unigram_model.char_dict))
        self.assertAlmostEqual(test_sentence_op[0], total_prob_exp, places=2)
        self.assertEqual(single_prob_exp, test_sentence_op[1])
        self.assertEqual(cumul_prob_exp, test_sentence_op[2])

    def test_unigram_test_str_untrained(self):
        unigram_model = UnigramModel(smoothing=0.0)
        unigram_model_smooth = UnigramModel()

        test_sentence_op = unigram_model.get_string_prob(UnigramTests.input_str)
        test_sentence_op_smooth = unigram_model_smooth.get_string_prob(UnigramTests.input_str)

        self.assertAlmostEqual(0.0, test_sentence_op[0], places=2)
        self.assertAlmostEqual(0.0, test_sentence_op_smooth[0], places=2)
        self.assertEqual({}, test_sentence_op[1])
        self.assertEqual({}, test_sentence_op_smooth[1])
        self.assertEqual([], test_sentence_op[2])
        self.assertEqual([], test_sentence_op_smooth[2])

    # Useful functions
    @staticmethod
    def str_unique_chars(input_str):
        return ''.join(set(input_str))


if __name__ == '__main__':
    unittest.main()
