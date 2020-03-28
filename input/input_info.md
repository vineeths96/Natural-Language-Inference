### Important information for input data

Please follow the steps listed below before exectuing the python files. 
This is to set up the input data properly.

1. Download the SNLI data corpus from the official link (http://nlp.stanford.edu/projects/snli/).
1. Extract the zip file (SNLI_1.0.zip) and obtain the data files.
1. Put the files _"snli_1.0_train.jsonl"_ and _"snli_1.0_test.jsonl"_ files in this folder.
1. Run _generate_meta_input()_ function from utils package (uncomment the function call in _main.py_) to generate pickle files of lists of cleaned and tokenized sentences 
1. The resulting pickle files are stored under './input/data_pickles/'
1. You are good to go.


### Important information for embedding data
1. Create a directory for embeddings at './input/embeddings/'
1. Download the GloVe embeddings from the official link (https://nlp.stanford.edu/projects/glove/)
1. Extract the zip file and put the GloVe files at the embeddings directory.
1. You are good to go.