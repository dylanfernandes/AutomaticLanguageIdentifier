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
        self.calculateProbablities()


    def gatherData(self, input_str):
        current = None
        #Stores dictionary of characters following current category
        occDict = {}
        testString = "aaaabbcdef"
        if input_str:
            self.training_size = len(input_str)
            #dictionary built from perspective of last character
            for nextChar in testString:
                #Skip first char
                if current != None:
                    if current in self.char_dict:
                        occDict = self.char_dict[current]
                        if nextChar in occDict:
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

                #set current char to previous before going to next char
                current = nextChar

            #no next character, just increment total
            if nextChar in self.char_dict:
                self.char_dict[nextChar]["total"] += 1
            else:
                self.char_dict[nextChar] = {"total":1}

        ##TODO check if missing chars, add to occurences if missing
					
    def calculateProbablities(self):
        prob = 0
        #Stores dictionary of characters following current category with probability for each
        probDict = {}
        for char in self.char_dict:
            for nextChar in self.char_dict[char]:
                if nextChar != "total":
                    #calculate probability
                    prob = self.char_dict[char][nextChar]/self.char_dict[char]["total"]
                    #get dictionary
                    if char in self.probs:
                        probDict = self.probs[char]
                    else:
                        probDict = {}
                    #add probability for nextChar in dictionary
                    probDict[nextChar] = prob
                    #update dictionary
                    self.probs[char] = probDict

        for char in self.probs:
            print(str(char) + ": " + str(self.probs[char]))
			
