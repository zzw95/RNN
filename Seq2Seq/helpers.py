import numpy as np
import os

def batch(inputs, max_sequence_length=None):
    """
    :param inputs: list of sentences (integer lists)
    :param max_sequence_length: integer specifying how large should `max_time` dimension be.
                                If None, maximum sequence length would be used
    :return: inputs_time_major: input sentences transformed into time-major matrix
                                (shape [max_time, batch_size]) padded with 0s
             sequence_lengths: batch-sized list of integers specifying amount of active time steps in each input sequence
    """
    sequence_lengths = [len(seq) for seq in inputs]
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    batch_size = len(inputs)
    inputs_time_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)

    for i,seq in enumerate(inputs):
        for j,word in enumerate(seq):
            inputs_time_major[i,j]=word

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_time_major.swapaxes(0,1)
    return inputs_time_major, sequence_lengths

def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    """
        Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
        raise ValueError('length_from > length_to')
    def random_length():
        if length_from==length_to:
            return length_from
        else:
            return np.random.randint(length_from,length_to+1)
    while True:
        yield [np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist()
           for _ in range(batch_size)]




def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()

    return data


def extract_vocab(data):
    special_words = ['<pad>', '<unk>', '<s>',  '<\s>']
    set_words = set([word for line in data.split('\n') for word in line.split()])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


def pad_id_sequences(source_ids, source_vocab_to_int, target_ids, target_vocab_to_int, sequence_length):
    new_source_ids = [list(reversed(sentence + [source_vocab_to_int['<pad>']] * (sequence_length - len(sentence)))) \
                      for sentence in source_ids]
    new_target_ids = [sentence + [target_vocab_to_int['<pad>']] * (sequence_length - len(sentence)) \
                      for sentence in target_ids]

    return new_source_ids, new_target_ids


def batch_data(source, target, batch_size):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield source_batch, target_batch
