import json
import spacy
import pickle
import unidecode
from word2number import w2n
from bs4 import BeautifulSoup
import contractions


def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    html_free = soup.get_text(separator=' ')

    return html_free


def remove_whitespace(text):
    text = text.strip()
    white_text = " ".join(text.split())

    return white_text


def lowercase(text):
    lower_text = text.lower()

    return lower_text


def remove_accented_char(text):
    accent_free = unidecode.unidecode(text)

    return accent_free


def expand_contractions(text):
    contraction_free = contractions.fix(text)

    return contraction_free


def preprocess(text, remove_htmltags=True, remove_extra_whitespace=True,
               remove_accent=True, remove_contractions=True,
               convert_lowercase=True, stop_words=True, punctuations=True,
               special_chars=True, remove_num=True, convert_num=True,
               lemmatization=True):
    if remove_htmltags:
        text = remove_html(text)

    if remove_extra_whitespace:
        text = remove_whitespace(text)

    if remove_accent:
        text = remove_accented_char(text)

    if remove_contractions:
        text = expand_contractions(text)

    if convert_lowercase:
        text = lowercase(text)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    cleaned_text = []

    for token in doc:
        flag = True
        token_text = token.text

        if stop_words and token.is_stop and token.pos_ != 'NUM':
            flag = False

        if punctuations and token.pos_ == 'PUNCT' and flag:
            flag = False

        if special_chars and token.pos_ == 'SYM' and flag:
            flag = False

        if remove_num and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag:
            flag = False

        if convert_num and token.pos_ == 'NUM' and flag:
            token_text = w2n.word_to_num(token.text)

        elif lemmatization and token.lemma_ != "-PRON-" and flag:
            token_text = token.lemma_

        if token_text != "" and flag:
            cleaned_text.append(token_text)

    return cleaned_text


def parse_input(fname, mode):
    file = open(fname)

    sentence1 = []
    sentence2 = []
    gold_label = []

    for line in file:
        data = json.loads(line)

        line_sentence1 = data['sentence1']
        line_sentence2 = data['sentence2']
        line_gold_label = data['gold_label']

        preprocessed_sentence1 = preprocess(line_sentence1, remove_num=False)
        preprocessed_sentence2 = preprocess(line_sentence2, remove_num=False)

        sentence1.append(preprocessed_sentence1)
        sentence2.append(preprocessed_sentence2)
        gold_label.append(line_gold_label)

    with open('./input/' + mode + '_list_sentence1.txt', "wb") as file:
        pickle.dump(sentence1, file)

    with open('./input/' + mode + '_list_sentence2.txt', "wb") as file:
        pickle.dump(sentence2, file)
