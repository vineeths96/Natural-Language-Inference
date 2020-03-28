import os
import json
import spacy
import pickle
import unidecode
import contractions
from word2number import w2n
from bs4 import BeautifulSoup

nlp = spacy.load('en_core_web_sm')


# Removes HTML tags from the text
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    html_free = soup.get_text(separator=' ')

    return html_free


# Removes extra whitespaces from the text
def remove_whitespace(text):
    text = text.strip()
    white_text = " ".join(text.split())

    return white_text


# Converts the text to lowercase
def lowercase(text):
    lower_text = text.lower()

    return lower_text


# Removes accented characters from the text
def remove_accented_char(text):
    accent_free = unidecode.unidecode(text)

    return accent_free


# Expands the contractions within the text
def expand_contractions(text):
    contraction_free = contractions.fix(text)

    return contraction_free


# Processes and cleans the input text as specified by the arguments
def preprocess(text, remove_htmltags=True, remove_extra_whitespace=True,
               remove_accent=True, remove_contractions=True,
               convert_lowercase=True, stop_words=True, punctuations=True,
               special_chars=True, remove_num=True, convert_num=True,
               lemmatization=True):
    # Call the necessary functions to perform cleaning
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

    # Use Spacy nlp() to tokenize the text
    doc = nlp(text)

    cleaned_text = []

    # CHeck whether each token belongs to any of the category to be removed,
    # which are specified by the function arguments
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

        try:
            if convert_num and token.pos_ == 'NUM' and flag:
                token_text = w2n.word_to_num(token.text)
        except:
            pass

        if lemmatization and token.lemma_ != "-PRON-" and flag:
            token_text = token.lemma_

        # If flag is True, which means that the token does not belong to any category
        # to be removed, we append it to the cleaned text list.
        if token_text != "" and flag:
            cleaned_text.append(token_text)

    return cleaned_text


# Receives a filename, processes it and dumps the processed lists to pickle files
def parse_input(fname, mode):
    try:
        file = open(fname)
    except:
        print("Files not found. Check the input folder (Ensure steps in input_info.md is followed)")
        exit(0)

    sentence1 = []
    sentence2 = []
    gold_label = []

    for line in file:
        # Loads the josn line into a dictionary
        data = json.loads(line)

        # Extract the necessary information from dictionary
        line_sentence1 = data['sentence1']
        line_sentence2 = data['sentence2']
        line_gold_label = data['gold_label']

        # Preprocess the sentences
        preprocessed_sentence1 = preprocess(line_sentence1, remove_num=False)
        preprocessed_sentence2 = preprocess(line_sentence2, remove_num=False)

        # Append the processed sentences and labels to a list
        sentence1.append(preprocessed_sentence1)
        sentence2.append(preprocessed_sentence2)
        gold_label.append(line_gold_label)

    # Store the processed lists as pickle files
    with open('./input/data_pickles/' + mode + '_list_sentence1.txt', "wb") as file:
        pickle.dump(sentence1, file)

    with open('./input/data_pickles/' + mode + '_list_sentence2.txt', "wb") as file:
        pickle.dump(sentence2, file)

    with open('./input/data_pickles/' + mode + '_list_gold_label.txt', "wb") as file:
        pickle.dump(gold_label, file)


# Master function to preprocess the train and test data files
def generate_meta_input():
    try:
        os.makedirs('./input/data_pickles')
    except:
        pass

    parse_input('./input/snli_1.0_train.jsonl', "train")
    parse_input('./input/snli_1.0_test.jsonl', "test")
