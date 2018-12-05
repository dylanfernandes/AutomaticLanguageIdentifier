import sys
sys.path.insert(0, '../')
from Code.main_script import *

OUTPUT_DIR_NO_LA = '../Output/no_latin_expr/'
OUTPUT_DIR_LA = '../Output/latin_expr/'
LATIN_SENTENCES_FILE = DATA_PATH + 'latin_sentences.txt'

TRAINING_FILES_NO_LATIN = {
    'en': ['trainEN.txt'],
    'fr': ['trainFR.txt'],
    'ot': ['trainOT.txt']
}

TRAINING_FILES_WITH_LATIN = {
    'en': ['trainEN.txt'],
    'fr': ['trainFR.txt'],
    'ot': ['trainOT.txt'],
    'la': ['trainLA.txt']
}


def main():
    # Without Latin training first
    print('Parsing Latin sentences without training a latin model:')
    exec_lang_parser(TRAINING_FILES_NO_LATIN, LATIN_SENTENCES_FILE, OUTPUT_DIR_NO_LA)
    print('Parsing without Latin training complete!')

    print("\n-----------------------------------------------------------------------\n")

    # With Latin training first
    print('Parsing Latin sentences with training a latin model:')
    exec_lang_parser(TRAINING_FILES_WITH_LATIN, LATIN_SENTENCES_FILE, OUTPUT_DIR_LA)
    print('Parsing with Latin training complete!')


if __name__ == '__main__':
    main()