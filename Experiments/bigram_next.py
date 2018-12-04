import pickle
import math

class BigramModelNext:
    smoothing = 0.5
    LOGBASE = 10
    trained = False
    smooth = False
    computeProb = False
    testComputed = False
    bins = 0

    def __init__(self, input_str=None, user_smoothing = 0.5):
        self.char_dict = {}
        self.probs = {}
        self.training_size = 0
        self.smoothing = user_smoothing
        if input_str:
            self.train(input_str)

    def train(self, input_str):
        self.gatherData(input_str)
        self.trained = True

    def test(self, string):
        self.smooth_char_dict(string)
        self.calculateProbablities()
        return self.get_string_prob(string)

    def gatherData(self, input_str):
        current = None
        #Stores dictionary of characters following current category
        occDict = {}
        if input_str:
            self.training_size = len(input_str)
            #dictionary built from perspective of last character
            for nextChar in input_str:
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
					
    def calculateProbablities(self):
        prob = 0
        #Stores dictionary of characters following current category with probability for each
        probDict = {}
        if self.trained:
            self.bins = len(self.char_dict.keys())
            for char in self.char_dict:
                for nextChar in self.char_dict[char]:
                    if nextChar != "total":
                        #calculate probability
                        if self.char_dict[char]["total"] > 0 or  self.smoothing > 0: 
                            prob = (self.char_dict[char][nextChar] + self.smoothing)/(self.char_dict[char]["total"] + (self.smoothing*self.bins))
                        else:
                            prob = 0
                        #get dictionary
                        if char in self.probs:
                            probDict = self.probs[char]
                        else:
                            probDict = {}
                        #add probability for nextChar in dictionary
                        probDict[nextChar] = prob
                        #update dictionary
                        self.probs[char] = probDict
            self.computeProb = True
        else:
            print("Training needs to be done before calculating probabilities")
			
    def smooth_char_dict(self, string):
        current = None
        #Stores dictionary of characters following current category
        occDict = {}
        #check all cases in char_dict
        if self.trained:
            for nextChar in string:
                if current != None:
                    #char never seen in training
                    if current not in self.char_dict:
                        occDict = {}
                        #will be used for smoothing
                        occDict["total"] = 0
                        occDict[nextChar] = 0
                        self.char_dict[current] = occDict
                    #nextChar never followed current in training set
                    if nextChar not in self.char_dict[current]:
                        occDict = self.char_dict[current]
                        #will be used for smoothing
                        occDict[nextChar] = 0
                current = nextChar
            self.smooth = True
        else:
            print("Training needs to be done before smoothing")
    
    def get_string_prob(self, string):
        total_prob = 0
        current = None
        result_cumul = {}
        result_single = {}
        bigramNum = 0
        if self.trained and self.computeProb:
            for nextChar in string:
                if current != None:
                    current_prob = math.log(self.probs[current][nextChar])/math.log(self.LOGBASE)
                    total_prob += current_prob
                    result_cumul[bigramNum] = {current + nextChar : total_prob}
                    result_single[bigramNum] = {current + nextChar : current_prob}
                    bigramNum += 1
                current = nextChar
                self.testComputed = True
        else:
            print("Training and computing probabilities needs to be done before testing a string")
        return [total_prob, result_single, result_cumul]