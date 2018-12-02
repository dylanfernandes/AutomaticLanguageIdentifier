import pickle

# TODO: when testing, try to load file from pickle, train if file does not exist.
class UnigramModel:
	
	SMOOTHING = 0.5
	VOCAB_SIZE = 26
	
	def __init__(self, input_str=None):
		self.char_dict = {}
		self.probs = {}
		self.training_size = 0
		if input_str:
			self.train(input)
	
	def train(self, input_str):
		# TODO: save to pickle
		if input_str:
			self.training_size = len(input_str)
			for char in input:
				if char in self.char_dict:
					self.char_dict[char] += 1
				else:
					self.char_dict[char] = 1
					
			for char in self.char_dict:
				self.probs[char] = self.calc_prob(char_count=self.char_dict[char], total_count=self.training_size)
	
	def calc_prob(self, char_count=0, smoothing=UnigramModel.SMOOTHING, total_count=0, vocab_size=UnigramModel.VOCAB_SIZE):
		return (char_count + smoothing) / (total_count + smoothing * vocab_size)
	
				
