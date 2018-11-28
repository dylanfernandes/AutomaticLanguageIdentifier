import sys
import re

DATA_PATH = '../DataSets/'
EN_FILES = ['en-the-little-prince.txt', 'en-moby-dick.txt']
FR_FILES = ['fr-le-petit-prince.txt', 'fr-vingt-mille-lieues-sous-les-mers.txt']
LANGUAGES = ['en', 'fr']
def main():
	print(load_file(DATA_PATH + LANGUAGES[0] +"/" + EN_FILES[0]))

#Returns cleaned content of file
def load_file(filePath):
	with open(filePath, 'r') as myfile:
		content=myfile.read().replace('\n', '')
	return re.sub("[^a-z]","", content.lower())



if __name__ == '__main__':
	main()
