import sys
sys.path.insert(0, '../Code')
import unittest
from bigram import BigramModel

class TestBigram(unittest.TestCase):
    def testTrain(self):
        testData = BigramModel("aaaabbcdef")
        self.assertEqual(testData.trained, True)

if __name__ == '__main__':
        unittest.main()