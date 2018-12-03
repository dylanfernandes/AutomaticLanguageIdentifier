import sys
sys.path.insert(0, '../Code')
import re

from bigram import BigramModel

DATA_PATH = '../DataSets/'
LANGUAGES = {
'en': ['en-the-little-prince.txt', 'en-moby-dick.txt'], 
'fr': ['fr-le-petit-prince.txt', 'fr-vingt-mille-lieues-sous-les-mers.txt']
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
	test_models()

def train_models():
	for language, documents in LANGUAGES.items():
		for document in documents:
			text = load_file(DATA_PATH + language +"/" + document)
			bigram = BigramModel(text, 0.5)
		bigrams[language] = bigram

def test_models():
	for language, bigram in bigrams.items():
		for sentence in SENTENCES:
			print(str(language) + " : " + str(sentence))
			text = clean_string(sentence)
			print(bigram.test(text))

def output_results():
	for sentence in SENTENCES:
		print(sentence + "\n")
		#unigram models goes here
		print("---------------- ")
		print("BIGRAM MODEL: ")



#Returns cleaned content of file
def load_file(filePath):
	with open(filePath, 'r', encoding="utf8", errors='ignore') as myfile:
		content=myfile.read()
	return clean_string(content)

def clean_string(string):
	return re.sub("[^a-z]","", string.lower().replace('\n', ''))

if __name__ == '__main__':
	main()
