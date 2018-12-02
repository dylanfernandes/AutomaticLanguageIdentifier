import sys
sys.path.insert(0, '../Experiments')
import unittest
import math

from bigram_next import BigramModelNext

class TestBigram(unittest.TestCase):
    def testTrain(self):
        testData = BigramModelNext("aaaabbcdef")
        self.assertEqual(testData.trained, True)
        self.assertEqual(len(testData.char_dict.keys()), 6)
        self.assertEqual(testData.char_dict["a"]["total"], 4)
        self.assertEqual(testData.char_dict["a"]["a"], 3)
        self.assertEqual(testData.char_dict["a"]["b"], 1)
        self.assertEqual(testData.char_dict["b"]["total"], 2)
        self.assertEqual(testData.char_dict["b"]["b"], 1)
        self.assertEqual(testData.char_dict["b"]["c"], 1)
        self.assertEqual(testData.char_dict["c"]["total"], 1)
        self.assertEqual(testData.char_dict["c"]["d"], 1)
        self.assertEqual(testData.char_dict["d"]["total"], 1)
        self.assertEqual(testData.char_dict["d"]["e"], 1)
        self.assertEqual(testData.char_dict["e"]["total"], 1)
        self.assertEqual(testData.char_dict["e"]["f"], 1)
        self.assertEqual(testData.char_dict["f"]["total"], 1)

        self.assertEqual(len(testData.char_dict["f"].keys()), 1)
        self.assertEqual("e" in testData.char_dict["f"], False)
        self.assertEqual("c" in testData.char_dict["a"], False)
    
    def testSmoothing(self):
        testData = BigramModelNext("aaaabbcdef")
        smooth = testData.SMOOTHING
        testData.smooth_char_dict("aagbef")
        self.assertEqual(len(testData.char_dict.keys()), 7)
        self.assertEqual(testData.char_dict["a"]["g"], smooth)
        self.assertEqual(testData.char_dict["a"]["total"], 4 + smooth)
        self.assertEqual(testData.char_dict["g"]["total"], smooth)
        self.assertEqual(testData.char_dict["g"]["b"], smooth)
        self.assertEqual(len(testData.char_dict["g"].keys()), 2)
        self.assertEqual(testData.char_dict["b"]["e"], smooth)
        self.assertEqual(testData.char_dict["b"]["total"], 2 + smooth)

        self.assertEqual(testData.char_dict["e"]["total"], 1)
        self.assertEqual(testData.char_dict["e"]["f"], 1)
        self.assertEqual(testData.char_dict["f"]["total"], 1)

    def testCalculateProbabilities(self):
        testData = BigramModelNext("aaaabbcdef")
        smooth = testData.SMOOTHING
        testData.smooth_char_dict("aagbef")
        testData.calculateProbablities()
        self.assertEqual(len(testData.probs.keys()), 6)
        #last char preceding end of string neglected
        self.assertEqual("f" in testData.probs, False)
        self.assertEqual(testData.probs["a"]["a"], 3/(4+smooth))
        self.assertEqual(testData.probs["a"]["g"], smooth/(4+smooth))
        self.assertEqual(testData.probs["g"]["b"], 1)
        self.assertEqual(testData.probs["b"]["e"], smooth/(2+smooth))
        self.assertEqual(testData.probs["e"]["f"], 1)
    
    def testGetStringProb(self):
        testData = BigramModelNext("aaaabbcdef")
        smooth = testData.SMOOTHING
        testData.smooth_char_dict("aagbef")
        testData.calculateProbablities()
        val = math.log10(3/(4+smooth)) + math.log10(smooth/(4+smooth)) + math.log10(1) + math.log10(smooth/(2+smooth)) + math.log10(1)
        val = -1 * val
        self.assertEqual(testData.get_string_prob("aagbef"), val)


if __name__ == '__main__':
        unittest.main()