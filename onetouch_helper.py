import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    music_examples = list(open("./data/onetouch/music.txt", "r").readlines())
    music_examples = [s.strip() for s in music_examples]

    call_examples = list(open("./data/onetouch/call.txt", "r").readlines())
    call_examples = [s.strip() for s in call_examples]

    command_examples = list(open("./data/onetouch/command.txt", "r").readlines())
    command_examples = [s.strip() for s in command_examples]

    message_examples = list(open("./data/onetouch/message.txt","r").readline())
    message_examples = [s.strip() for s in message_examples]

    navigation_examples = list(open("./data/onetouch/navigation.txt","r").readline())
    navigation_examples = [s.strip() for s in navigation_examples]

    parking_examples = list(open("./data/onetouch/parking.txt","r").readline())
    parking_examples = [s.strip() for s in parking_examples]


    # Split by words
    x_text = music_examples + call_examples + command_examples + \
             message_examples + navigation_examples + parking_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    music_labels = [[1, 0, 0,0,0,0] for _ in music_examples]
    call_labels = [[0, 1, 0,0,0,0] for _ in call_examples]
    command_labels = [[0,0,1,0,0,0] for _ in command_examples]
    message_labels = [[0,0,0,1,0,0] for _ in message_examples]
    navigation_labels = [[0,0,0,0,1,0] for _ in navigation_examples]
    parking_labels = [[0,0,0,0,0,1] for _ in parking_examples]

    y = np.concatenate([music_labels, call_labels, command_labels,message_labels,navigation_labels,parking_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]