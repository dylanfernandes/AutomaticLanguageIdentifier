import sys
import re

DATA_PATH = '../DataSets/'
LANGUAGES = {
'en': ['en-the-little-prince.txt', 'en-moby-dick.txt'], 
'fr': ['fr-le-petit-prince.txt', 'fr-vingt-mille-lieues-sous-les-mers.txt']
}

def main():
	for language, documents in LANGUAGES.items():
		for document in documents:
			text = load_file(DATA_PATH + language +"/" + document)
			string = clean_string("What will the Japanese economy be like next year?")
			#Call models to get probability
			break
		break

#Returns cleaned content of file
def load_file(filePath):
	with open(filePath, 'r', encoding="utf8", errors='ignore') as myfile:
		content=myfile.read()
	return clean_string(content)

def clean_string(string):
	return re.sub("[^a-z]","", string.lower().replace('\n', ''))

if __name__ == '__main__':
	main()
