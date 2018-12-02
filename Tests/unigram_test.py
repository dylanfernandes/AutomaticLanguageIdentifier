import unittest
from Code.unigram import UnigramModel
from math import log10


class UnigramTests(unittest.TestCase):

    train_str = "zahhjewjkojfcfdssd"
    input_str = "aaabbcdddd"

    def test_train(self):

        train_str_unique = UnigramTests.str_unique_chars(UnigramTests.train_str)
        unigram_model = UnigramModel(UnigramTests.train_str, False)
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
        unigram_model = UnigramModel(UnigramTests.train_str)

        # test one of the probability calculations
        # For character 'z'
        prob_test = log10((1.0 + UnigramModel.SMOOTHING) /
                          (len(UnigramTests.train_str) + UnigramModel.SMOOTHING * UnigramModel.VOCAB_SIZE))
        self.assertEqual(unigram_model.probs['z'], prob_test)

    def test_multi_doc_train(self):

        train_str_unique = UnigramTests.str_unique_chars(UnigramTests.train_str)
        mid_str = int(len(UnigramTests.train_str) / 2)
        unigram_model = UnigramModel([UnigramTests.train_str[:mid_str], UnigramTests.train_str[mid_str:]])

        self.assertEqual(len(unigram_model.char_dict), len(train_str_unique))
        self.assertEqual(len(unigram_model.probs), len(unigram_model.char_dict))

        total_count = 0
        for char_count in unigram_model.char_dict.values():
            total_count += char_count

        self.assertEqual(total_count, len(UnigramTests.train_str))

        self.assertTrue(unigram_model.trained)

    def test_unigram_test_str(self):
        unigram_model = UnigramModel(UnigramTests.train_str, False)

        # Manually calculated
        expected_val = -8.838
        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str), expected_val, places=2)

    def test_unigram_test_str_smooth(self):
        unigram_model = UnigramModel(UnigramTests.train_str)

        # Manually calculated
        expected_val = -13.22
        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str), expected_val, places=2)

    def test_multi_doc_test_str(self):
        mid_str = int(len(UnigramTests.train_str) / 2)
        unigram_model = UnigramModel([UnigramTests.train_str[:mid_str], UnigramTests.train_str[mid_str:]])

        # Manually calculated
        expected_val = -13.22
        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str), expected_val, places=2)

    def test_unigram_test_str_untrained(self):
        unigram_model = UnigramModel(smooth=False)
        unigram_model_smooth = UnigramModel()

        self.assertAlmostEqual(unigram_model.prob_sentence(UnigramTests.input_str), 0.0, places=2)
        self.assertAlmostEqual(unigram_model_smooth.prob_sentence(UnigramTests.input_str), 0.0, places=2)

    # Useful functions
    @staticmethod
    def str_unique_chars(input_str):
        return ''.join(set(input_str))


if __name__ == '__main__':
    unittest.main()
