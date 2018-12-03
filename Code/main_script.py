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

bigrams = {}

def main():
	train_models()
	output_results()

def train_models():
	for language, documents in TRAINING_FILES.items():
		for document in documents:
			text = load_file(DATA_PATH + language +"/" + document)
			bigram = BigramModel(text, 0.5)
		bigrams[language] = bigram

def test_models():
	for language, bigram in bigrams.items():
		for sentence in SENTENCES:
			print(str(language) + " : " + str(sentence))
			text = clean_string(sentence)
			print(bigram.test(text)[1])

def output_results():
	orig_stdout = sys.stdout
	output_file_template = "../Output/out"
	sentence_num = 1
	#dictionary of results from testing sentences
	result = {}
	for sentence in SENTENCES:
		writer = open(output_file_template+str(sentence_num)+'.txt', 'w')
		sys.stdout = writer
		print(sentence + "\n")
		#unigram models goes here
		print("---------------- ")
		print("BIGRAM MODEL: ")
		text = clean_string(sentence)
		for language, bigram in bigrams.items():
			result[language] = bigram.test(text)[0]
		for i in range(len(text)-1):
			for language, result_array in result.items():
				key = list(result_array[i].keys())[0]
				prob = result_array[i][key]
				print (LANGUAGES[language] + ": P("+ str(key[1]) + "|" +  str(key[0]) + ") = " + str(prob))
		sys.stdout = orig_stdout
		writer.close()
		sentence_num += 1


#Returns cleaned content of file
def load_file(filePath):
	with open(filePath, 'r', encoding="utf8", errors='ignore') as myfile:
		content=myfile.read()
	return clean_string(content)

def clean_string(string):
	return re.sub("[^a-z]","", string.lower().replace('\n', ''))

if __name__ == '__main__':
	main()
