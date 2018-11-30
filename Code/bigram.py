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
        self.gatherData(input_str)
        for char in self.char_dict:
            print(str(char) + ": " + str(self.char_dict[char]))
            #for char in self.prev_dict[prev]:
                #print(char)
                #self.probs[prev] = self.calc_prob(char_count=self.char_dict[char], total_count=self.training_size)


    def gatherData(self, input_str):
        current = None
        #Stores dictionary of characters following current category
        occDict = {}
        testString = "aaaabbcdef"
        if input_str:
            self.training_size = len(input_str)
            #dictionary built from perspective of last character
            for nextChar in input_str:
                #Skip first char
                if current != None:
                    if current in self.char_dict:
                        occDict = self.char_dict[current]
                        #print(str(prev) + "||" + str(occDict))
                        if nextChar in occDict:
                            #print("Seen char")
                            occDict[nextChar] += 1
                        else:
                            occDict[nextChar] = 1
                        occDict["total"] += 1
                        self.char_dict[current] = occDict
                    else:
                        #create dictionary for new char
                        occDict = {}
                        occDict[nextChar] = 1
                        occDict["total"] = 1
                        self.char_dict[current] = occDict
                        #print(str(current) + "||" + str(self.char_dict[current]))
                #set current char to previous
                current = nextChar

            #no next character, just increment total
            if nextChar in self.char_dict:
                self.char_dict[nextChar]["total"] += 1
            else:
                self.char_dict[nextChar] = {"total":1}
					
	
    #def calc_prob(self, char_count=0, smoothing=BigramModel.SMOOTHING, total_count=0, vocab_size=BigramModel.VOCAB_SIZE):
		#return (char_count + smoothing) / (total_count + smoothing * vocab_size)
			
