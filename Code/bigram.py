import pickle
import math
import sys

class BigramModel:
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
        previous = None
        #Stores dictionary of characters following current category
        occDict = {}
        if input_str:
            self.training_size = len(input_str)
            #dictionary built from perspective of last character
            for current in input_str:
                #Skip first char
                if previous != None:
                    if current in self.char_dict:
                        occDict = self.char_dict[current]
                        if previous in occDict:
                            occDict[previous] += 1
                        else:
                            occDict[previous] = 1
                        occDict["total"] += 1
                        self.char_dict[current] = occDict
                    else:
                        #create dictionary for new char
                        occDict = {}
                        occDict[previous] = 1
                        occDict["total"] = 1
                        self.char_dict[current] = occDict
                else:
                    #add occurence for first character of string
                    occDict = {}
                    occDict["total"] = 1
                    self.char_dict[current] = occDict


                #set current char to previous before going to next char
                previous = current
					
    def calculateProbablities(self):
        prob = 0
        #Stores dictionary of characters following current category with probability for each
        probDict = {}
        if self.trained:
            self.bins = len(self.char_dict.keys())
            for char in self.char_dict:
                for previous in self.char_dict[char]:
                    if previous != "total":
                        #calculate probability
                        if self.char_dict[char]["total"] > 0 or  self.smoothing > 0: 
                            prob = (self.char_dict[char][previous] + self.smoothing)/(self.char_dict[char]["total"] + (self.smoothing*self.bins))
                        else:
                            prob = 0
                        #get dictionary
                        if char in self.probs:
                            probDict = self.probs[char]
                        else:
                            probDict = {}
                        #add probability for nextChar in dictionary
                        probDict[previous] = prob
                        #update dictionary
                        self.probs[char] = probDict
            self.computeProb = True
        else:
            print("Training needs to be done before calculating probabilities")
			
    def smooth_char_dict(self, string):
        previous = None
        #Stores dictionary of characters following current category
        occDict = {}
        #check all cases in char_dict
        if self.trained:
            for current in string:
                if previous != None:
                    #char never seen in training
                    if current not in self.char_dict:
                        occDict = {}
                        #will be used for smoothing
                        occDict["total"] = 0
                        occDict[previous] = 0
                        self.char_dict[current] = occDict
                    #nextChar never followed current in training set
                    if previous not in self.char_dict[current]:
                        occDict = self.char_dict[current]
                        #will be used for smoothing
                        occDict[previous] = 0
                previous = current
            self.smooth = True
        else:
            print("Training needs to be done before smoothing")
    
    def get_string_prob(self, string):
        total_prob = 0
        previous = None
        result_cumul = {}
        result_single = {}
        bigramNum = 0
        if self.trained and self.computeProb:
            for current in string:
                if previous != None:
                    current_prob = math.log(self.probs[current][previous])/math.log(self.LOGBASE)
                    total_prob += current_prob
                    result_cumul[bigramNum] = {previous + current : total_prob}
                    result_single[bigramNum] = {previous + current : current_prob}
                    bigramNum += 1
                previous = current
            self.testComputed = True
        else:
            print("Training and computing probabilities needs to be done before testing a string")
        return [total_prob, result_single, result_cumul]

    def dump_probs(self, file):
        if self.testComputed:
            orig_stdout = sys.stdout
            writer = open(file, 'w')
            sys.stdout = writer
            for char in self.probs:
                for previous in self.probs[char]:
                    current_prob = math.log(self.probs[char][previous])/math.log(self.LOGBASE)
                    print("P(" + str(char) + "|" + str(previous) + ") = " + str(current_prob))
            sys.stdout = orig_stdout
            writer.close()
