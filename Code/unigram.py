


def unigram(char_dict, sentence):
		for char in sentence:
				char_dict[char] += 1

def bigram(char_2d_dict, sentence):
	for i in range(1, len(sentence)):
		
