import sys
sys.path.insert(0, '../Code')
import re

from bigram import BigramModel
from bigram_next import BigramModelNext

DATA_PATH = '../DataSets/'
TRAINING_FILES = {
'en': ['en-the-little-prince.txt', 'en-moby-dick.txt'], 
'fr': ['fr-le-petit-prince.txt', 'fr-vingt-mille-lieues-sous-les-mers.txt'],
'ot': ['sp-el-principito.txt', 'sp-moby-dick.txt']
}

LANGUAGES = {
	'en': 'ENGLISH', 
	'fr': "FRENCH",
	'ot': "OTHER"
}

SENTENCES = {
	"Boutique anglaise.",
	"At which cafe?",
	"Un bon chef",
	"What boutique?"
}


def main():
	trained_models = train_models()
	output_results(trained_models[0], trained_models[1])

def train_models():
    bigrams = {}
    next_bigrams = {}
    for language, documents in TRAINING_FILES.items():
        bigram = BigramModel(user_smoothing=0.5)
        next_bigram = BigramModelNext(user_smoothing = 0.5)
        for document in documents:
            text = load_file(DATA_PATH + language +"/" + document)
            bigram.train(text)
            next_bigram.train(text)
        bigrams[language] = bigram
        next_bigrams[language]= next_bigram
    return [bigrams, next_bigrams]

def output_results(bigrams, next_bigrams):
    orig_stdout = sys.stdout
    output_file_template = "../Output/BigramExperiment/out"
    sentence_num = 1
    #dictionary of results from testing sentences
    for sentence in SENTENCES:
        #change output location
        writer = open(output_file_template+str(sentence_num)+'.txt', 'w')
        sys.stdout = writer
        print(sentence + "\n")
        bigram_output(sentence, bigrams)
        bigram_output(sentence, next_bigrams, False)
        sentence_num += 1
        #close writer
        sys.stdout = orig_stdout
        writer.close()

def bigram_output(sentence, bigrams, prev = True):
    result_cumul = {}
    result_single = {}
    result_prob = {}
    char1 = None
    char2 = None
    print("---------------- ")
    if prev:
        print("BIGRAM MODEL: ")
    else:
        print("BIGRAM NEXT MODEL: ")
    text = clean_string(sentence)
    #Get and store test results
    for language, bigram in bigrams.items():
        results = bigram.test(text)
        result_prob[language] = results[0]
        result_single[language] = results[1]
        result_cumul[language] = results[2]
    #Output results according to format in project specs
    for i in range(len(text)-1):
        #For bigram printing
        j = 0
        for language, result_array in result_cumul.items():
            result_array_single = result_single[language]
            key = list(result_array[i].keys())[0]
            prob_single = result_array_single[i][key]
            prob_cumul = result_array[i][key]
            if j == 0:
                if prev:
                    print("Bigram: " + key)
                else:
                    print("Bigram Next: " + key)
            if prev:
                char1 = str(key[1]) 
                char2 = str(key[0])
            else:
                char1 = str(key[0]) 
                char2 = str(key[1])
            print (LANGUAGES[language] + ": P("+ char1 + "|" +  char2 + ") = " + str(prob_single) + " ==> log prob of sentence so far: " + str(prob_cumul))
            j += 1
        print()
    print("According to the bigram model, the sentence is in " + get_best_language(result_prob))

#find best language with total probabilities
def get_best_language(result_prob):
	best_prob = None
	best_lang = None
	for language in result_prob:
		prob = result_prob[language]
		if not best_prob:
			best_prob = prob
			best_lang = language
		elif prob > best_prob:
			best_prob = prob
			best_lang = language
	return LANGUAGES[best_lang]


#Returns cleaned content of file
def load_file(filePath):
	with open(filePath, 'r', encoding="utf8", errors='ignore') as myfile:
		content=myfile.read()
	return clean_string(content)

def clean_string(string):
	return re.sub("[^a-z]","", string.lower().replace('\n', ''))

if __name__ == '__main__':
	main()
