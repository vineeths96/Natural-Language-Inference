from utils.parse_input import parse_input


def read_input():
    parse_input('./input/snli_1.0_train.jsonl', "train")
    parse_input('./input/snli_1.0_test.jsonl', "test")