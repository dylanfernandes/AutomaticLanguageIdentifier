import sys
sys.path.insert(0, '../Code')
import re

from bigram import BigramModel

DATA_PATH = '../DataSets/'
TRAINING_FILES = {
'en': ['en-the-little-prince.txt', 'en-moby-dick.txt'], 
'fr': ['fr-le-petit-prince.txt', 'fr-vingt-mille-lieues-sous-les-mers.txt']
}

LANGUAGES = {
	'en': 'ENGLISH', 
	'fr': "FRENCH"
}
SENTENCES = {
	"What will the Japanese economy be like next year?",
	"She asked him if he was a student at this school.",
	"I'm OK.",
	"Birds build nests.",
	"I hate AI.",
	"L'oiseau vole.",
	"Woody Allen parle.",
	"Est-ce que l'arbitre est la?",
	"Cette phrase est en anglais.",
	"J'aime l'IA."
}


def main():
	bigrams = train_models()
	output_results(bigrams)

def train_models():
	bigrams = {}
	for language, documents in TRAINING_FILES.items():
		for document in documents:
			text = load_file(DATA_PATH + language +"/" + document)
			bigram = BigramModel(text, 0.5)
		bigrams[language] = bigram
	return bigrams

def output_results(bigrams):
	orig_stdout = sys.stdout
	output_file_template = "../Output/out"
	sentence_num = 1
	#dictionary of results from testing sentences
	for sentence in SENTENCES:
		#change output location
		writer = open(output_file_template+str(sentence_num)+'.txt', 'w')
		sys.stdout = writer
		print(sentence + "\n")
		#unigram models goes here
		bigram_output(sentence, bigrams)
		sentence_num += 1
		#close writer
		sys.stdout = orig_stdout
		writer.close()

def bigram_output(sentence, bigrams):
		result_cumul = {}
		result_single = {}
		result_prob = {}
		print("---------------- ")
		print("BIGRAM MODEL: ")
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
					print("Bigram: " + key)
				print (LANGUAGES[language] + ": P("+ str(key[1]) + "|" +  str(key[0]) + ") = " + str(prob_single) + " ==> log prob of sentence so far: " + str(prob_cumul))
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
