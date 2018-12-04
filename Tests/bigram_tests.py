import sys
sys.path.insert(0, '../Code')
import unittest
import math

from bigram import BigramModel

class TestBigram(unittest.TestCase):
    def testTrain(self):
        testData = BigramModel("aaaabbcdef")
        self.assertEqual(testData.trained, True)
        self.assertEqual(len(testData.char_dict.keys()), 6)
        self.assertEqual(testData.char_dict["a"]["total"], 4)
        self.assertEqual(testData.char_dict["a"]["a"], 3)
        self.assertEqual(testData.char_dict["b"]["total"], 2)
        self.assertEqual(testData.char_dict["b"]["a"], 1)
        self.assertEqual(testData.char_dict["b"]["b"], 1)
        self.assertEqual(testData.char_dict["c"]["total"], 1)
        self.assertEqual(testData.char_dict["c"]["b"], 1)
        self.assertEqual(testData.char_dict["d"]["total"], 1)
        self.assertEqual(testData.char_dict["d"]["c"], 1)
        self.assertEqual(testData.char_dict["e"]["total"], 1)
        self.assertEqual(testData.char_dict["e"]["d"], 1)
        self.assertEqual(testData.char_dict["f"]["total"], 1)
        self.assertEqual(testData.char_dict["f"]["e"], 1)

        self.assertEqual(len(testData.char_dict["f"].keys()), 2)
        self.assertEqual("e" in testData.char_dict["f"], True)
        self.assertEqual("c" in testData.char_dict["a"], False)
    
    def test_smoothing(self):
        testData = BigramModel("aaaabbcdef")
        testData.smooth_char_dict("aagbef")
        self.assertEqual(len(testData.char_dict.keys()), 7)
        self.assertEqual(testData.char_dict["g"]["a"], 0)
        self.assertEqual(testData.char_dict["a"]["total"], 4)
        self.assertEqual(testData.char_dict["g"]["total"], 0)
        self.assertEqual(testData.char_dict["b"]["g"], 0)
        self.assertEqual(len(testData.char_dict["g"].keys()), 2)
        self.assertEqual(testData.char_dict["e"]["b"], 0)
        self.assertEqual(testData.char_dict["b"]["total"], 2)
        self.assertEqual(testData.char_dict["e"]["d"], 1)
        self.assertEqual(testData.char_dict["e"]["total"], 1)
        self.assertEqual(testData.char_dict["f"]["e"], 1)
        self.assertEqual(testData.char_dict["f"]["total"], 1)

    def test_calculate_probabilities(self):
        testData = BigramModel("aaaabbcdef")
        smooth = testData.smoothing
        testData.smooth_char_dict("aagbef")
        testData.calculateProbablities()
        self.assertEqual(len(testData.probs.keys()), 7)
        self.assertEqual("f" in testData.probs, True)
        self.assertEqual(testData.probs["a"]["a"], (3+smooth)/(4+(smooth*7)))
        self.assertEqual(testData.probs["g"]["a"], smooth/(smooth*7))
        self.assertEqual(testData.probs["b"]["g"], smooth/(2+(smooth*7)))
        self.assertEqual(testData.probs["e"]["b"], smooth/(1+(smooth*7)))
        self.assertEqual(testData.probs["f"]["e"], (1+smooth)/(1+(smooth*7)))
    
    def test_get_string_prob(self):
        testData = BigramModel("aaaabbcdef")
        smooth = testData.smoothing
        testData.smooth_char_dict("aagbef")
        testData.calculateProbablities()
        results = testData.get_string_prob("aagbef")

        result_single = results[1]
        key = list(result_single[0].keys())[0]
        self.assertEqual(key, "aa")
        val = math.log10((3+smooth)/(4+(smooth*7)))
        self.assertAlmostEqual(result_single[0][key], val)

        key = list(result_single[1].keys())[0]
        self.assertEqual(key, "ag")
        val = math.log10(smooth/(smooth*7))
        self.assertAlmostEqual(result_single[1][key], val)

        key = list(result_single[2].keys())[0]
        self.assertEqual(key, "gb")
        val = math.log10(smooth/(2+(smooth*7)))
        self.assertAlmostEqual(result_single[2][key], val)

        key = list(result_single[3].keys())[0]
        self.assertEqual(key, "be")
        val = math.log10(smooth/(1+smooth*7))
        self.assertAlmostEqual(result_single[3][key], val)

        key = list(result_single[4].keys())[0]
        self.assertEqual(key, "ef")
        val = math.log10((1+smooth)/(1+(smooth*7)))
        self.assertAlmostEqual(result_single[4][key], val)

        val = 0
        result_cumul = results[2]
        key = list(result_cumul[0].keys())[0]
        self.assertEqual(key, "aa")
        val += math.log10((3+smooth)/(4+(smooth*7)))
        self.assertAlmostEqual(result_cumul[0][key], val)

        key = list(result_cumul[1].keys())[0]
        self.assertEqual(key, "ag")
        val += math.log10(smooth/(smooth*7))
        self.assertAlmostEqual(result_cumul[1][key], val)

        key = list(result_cumul[2].keys())[0]
        self.assertEqual(key, "gb")
        val += math.log10(smooth/(2+(smooth*7)))
        self.assertAlmostEqual(result_cumul[2][key], val)

        key = list(result_cumul[3].keys())[0]
        self.assertEqual(key, "be")
        val += math.log10(smooth/(1+smooth*7))
        self.assertAlmostEqual(result_cumul[3][key], val)

        key = list(result_cumul[4].keys())[0]
        self.assertEqual(key, "ef")
        val += math.log10((1+smooth)/(1+(smooth*7)))
        self.assertAlmostEqual(result_cumul[4][key], val)

        self.assertAlmostEqual(results[0], val)


if __name__ == '__main__':
        unittest.main()