### Important information for input data

Please follow the steps listed below before exectuing the python files. 
This is to set up the input data properly.

1. Download the SNLI data corpus from the official link (http://nlp.stanford.edu/projects/snli/).
1. Extract the zip file (SNLI_1.0.zip) and obtain the data files.
1. Put the files _"snli_1.0_train.jsonl"_ and _"snli_1.0_test.jsonl"_ files in this folder.
1. Run _generate_meta_input()_ function from utils package to generate pickle files (list) of cleaned and tokenized sentences 
1. You are good to go.