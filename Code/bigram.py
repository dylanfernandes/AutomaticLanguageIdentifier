import pickle

class BigramModel:
    SMOOTHING = 0.5
    VOCAB_SIZE = 26

    def __init__(self, input_str=None):
        self.char_dict = {}
        self.probs = {}
        self.training_size = 0
        if input_str:
            self.train(input_str)

    def train(self, input_str):
        prev = None
        prevDict = {}
        if input_str:
            self.training_size = len(input_str)
            for char in input_str:
                if prev == None:
                    prev = char
                else:
                    if char in self.char_dict:
                        prevDict = self.char_dict[char]
                        prevDict[prev] += 1
                        self.char_dict[char] = prevDict
                    else:
                        self.char_dict[char] = {char:1}
					
            #for char in self.char_dict:
				#self.probs[char] = self.calc_prob(char_count=self.char_dict[char], total_count=self.training_size)
	
    #def calc_prob(self, char_count=0, smoothing=BigramModel.SMOOTHING, total_count=0, vocab_size=BigramModel.VOCAB_SIZE):
		#return (char_count + smoothing) / (total_count + smoothing * vocab_size)