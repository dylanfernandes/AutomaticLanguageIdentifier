import sys
import re
from Code.unigram import UnigramModel

DATA_PATH = '../DataSets/'

LANGUAGES = {
    'en': ['en-the-little-prince.txt', 'en-moby-dick.txt'],
    'fr': ['fr-le-petit-prince.txt', 'fr-vingt-mille-lieues-sous-les-mers.txt']
}

UNIGRAM_MODELS = {
    'en': UnigramModel(),
    'fr': UnigramModel()
}

TEST_SENTENCES = [
    "What will the Japanese economy be like next year?",
    "She asked him if he was a student at this school.",
    "I'm OK.",
    "Birds build nests.",
    "I hate AI.",
    "L’oiseau vole.",
    "Woody Allen parle.",
    "Est-ce que l’arbitre est la?",
    "Cette phrase est en anglais.",
    "J’aime l’IA."
]


def main():
    exec_unigrams(TEST_SENTENCES)


# Returns cleaned content of file
def load_file(filePath):
    with open(filePath, 'r', encoding="utf8", errors='ignore') as myfile:
        content = myfile.read()
    return clean_string(content)


def clean_string(string):
    return re.sub("[^a-z]","", string.lower().replace('\n', ''))


def train_unigrams():

    for language, documents in LANGUAGES.items():

        train_docs = []
        for document in documents:
            train_docs.append(load_file(DATA_PATH + language + "/" + document))

        UNIGRAM_MODELS[language].train(train_docs)


def test_unigrams(test_str):
    clean_str = clean_string(test_str)
    highest_prob = None  # Is supposed to be a float, but set to none here for the first iteration's case
    detected_lang = ''

    for language, unigram in UNIGRAM_MODELS.items():
        prob = unigram.prob_sentence(clean_str)

        if not highest_prob:
            highest_prob = prob
            detected_lang = language
        elif highest_prob < prob:
            highest_prob = prob
            detected_lang = language

    return detected_lang


def exec_unigrams(test_sentences):
    train_unigrams()
    for test_sentence in test_sentences:
        detected_lang = test_unigrams(test_sentence)
        print('The following sentence:\n{0}\n...is most likely {1}!\n'.format(test_sentence, detected_lang))


if __name__ == '__main__':
    main()
