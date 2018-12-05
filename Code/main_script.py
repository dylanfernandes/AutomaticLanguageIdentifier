import sys
sys.path.insert(0, '../')
import re

from Code.bigram import BigramModel
from Code.unigram import UnigramModel

DATA_PATH = '../DataSets/'
OUTPUT_DIR = "../Output/main/"

TRAINING_FILES_1 = {
	'en': ['trainEN.txt'],
	'fr': ['trainFR.txt'],
	'ot': ['trainOT.txt']
}

TRAINING_FILES = {
	'en': ['en-the-little-prince.txt', 'en-moby-dick.txt'],
	'fr': ['fr-le-petit-prince.txt', 'fr-vingt-mille-lieues-sous-les-mers.txt'],
	'ot': ['sp-el-principito.txt', 'sp-moby-dick.txt']
}

LANGUAGES = {
	'en': 'ENGLISH', 
	'fr': "FRENCH",
	'ot': "SPANISH"
}

SENTENCES = [
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
]

SENTENCES_GOOD = [
	"Pido perdon a los ninos por haber dedicado este libro a una persona mayor.",
	"Mi dibujo no representaba un sombrero.",
	"De repente una enorme masa emergio del agua, y se disparoverticalmente por el aire.",
	"Es la imagen del inaferrable fantasma de la vida; y esa es la clave de todo ello.",
	"I jumped to my feet, completely thunderstruck.",
	"But my drawing is certainly very much less charming than its model.",
	"That, however, is not my fault.",
	"J'ai bien frotte mes yeux.",
	"C'est utile, si l'on est egare pendant la nuit.",
	"J'ai donc du choisir un autre metier et j'ai appris a piloter des avions."
]


def main():
	exec_lang_parser(TRAINING_FILES_1, (DATA_PATH + "sentences.txt"), OUTPUT_DIR)


def exec_lang_parser(training_files, sentences_file, output_filename):
	unigrams, bigrams = train_models(training_files)
	sentences = load_sentences(sentences_file)
	output_results(sentences, unigrams, bigrams, output_filename)


def train_models(training_files):
	unigrams = {}
	bigrams = {}

	for language, documents in training_files.items():
		unigram = UnigramModel(smoothing=0.5)
		bigram = BigramModel(user_smoothing=0.5)

		for document in documents:
			text = load_file(DATA_PATH + language + "/" + document)
			unigram.train(text)
			bigram.train(text)

		unigrams[language] = unigram
		bigrams[language] = bigram
	return unigrams, bigrams


def load_sentences(sentences_file):
	sentences = []
	with open(sentences_file, 'r') as file:
		# Loads all sentences omitting those that begin with
		# a '#' comment signal and omitting single newline characters
		for sentence in file.read().split('\n'):
			if len(sentence) > 0:
				if sentence[0] != '#':
					sentences.append(sentence)
	return sentences


def output_results(sentences, unigrams, bigrams, output_dir):
	orig_stdout = sys.stdout
	sentence_num = 1
	output_template = output_dir + "out"
	# Dictionary of results from testing sentences
	for sentence in sentences:

		# Change output location
		writer = open(output_template + str(sentence_num) + '.txt', 'w')
		sys.stdout = writer
		print(sentence + "\n")

		unigram_output(sentence, unigrams, output_dir)
		bigram_output(sentence, bigrams, output_dir)

		sentence_num += 1

		# Close writer
		sys.stdout = orig_stdout
		writer.close()


def unigram_output(sentence, unigrams, output_dir):

	results_prob = {}
	results_single = {}
	results_cumul = {}

	print("UNIGRAM MODEL: \n")
	text = clean_string(sentence)

	output_dir += "model/"
	# Get and store test results
	for language, unigram in unigrams.items():
		results = unigram.get_string_prob(text)
		unigram.dump_probs(output_dir + "unigram" + language.upper() + ".txt")
		results_prob[language] = results[0]
		results_single[language] = results[1]
		results_cumul[language] = results[2]

	# Output results according to project format
	for i in range(len(text)):

		print_char_title = True
		for language, result_cumul in results_cumul.items():
			char = result_cumul[i][0]
			cumul_prob = result_cumul[i][1]
			prob_single = results_single[language][char]

			if print_char_title:
				print("UNIGRAM: {}".format(char))
				print_char_title = False

			print("{language}: P({char}) = {prob_single} ==> log prob of sentence so far: {cumul_prob}"
				  .format(language=language, char=char, prob_single=prob_single, cumul_prob=cumul_prob))
		
		print()
	print("According to the unigram model, the sentence is in {}".format(get_best_language(results_prob)))


def bigram_output(sentence, bigrams, output_dir):
		result_cumul = {}
		result_single = {}
		result_prob = {}
		print("\n----------------\n")
		print("BIGRAM MODEL: \n")
		text = clean_string(sentence)

		output_dir += "model/"
		# Get and store test results
		for language, bigram in bigrams.items():
			results = bigram.test(text)
			bigram.dump_probs(output_dir + "bigram" + language.upper() + ".txt")
			result_prob[language] = results[0]
			result_single[language] = results[1]
			result_cumul[language] = results[2]
		# Output results according to format in project specs
		for i in range(len(text)-1):
			# For bigram printing
			j = 0
			for language, result_array in result_cumul.items():
				result_array_single = result_single[language]
				key = list(result_array[i].keys())[0]
				prob_single = result_array_single[i][key]
				prob_cumul = result_array[i][key]
				if j == 0:
					print("BIGRAM: " + key)
				print(LANGUAGES[language] + ": P(" + str(key[1]) + "|" +  str(key[0]) + ") = " + str(prob_single) + " ==> log prob of sentence so far: " + str(prob_cumul))
				j += 1
			print()
		print("According to the bigram model, the sentence is in " + get_best_language(result_prob))


# Find best language with total probabilities
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


# Returns cleaned content of file
def load_file(filePath):
	with open(filePath, 'r', encoding="utf8", errors='ignore') as myfile:
		content=myfile.read()
	return clean_string(content)


def clean_string(string):
	return re.sub("[^a-z]", "", string.lower().replace('\n', ''))



if __name__ == '__main__':
	main()
