from numpy import array
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re


def encode_me(seq):
    # get sequence into an array
    seq_array = array(list(seq))

    # integer encode the sequence
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array)

    # one hot the sequence
    onehot_encoder = OneHotEncoder(sparse=False)
    # reshape because that's what OneHotEncoder likes
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    onehot_encoded_seq = array(onehot_encoded_seq)
    padded_array = np.zeros((41, 4))
    print(seq)
    padded_array[0:len(seq), :] = onehot_encoded_seq
    return padded_array


def one_hot_encode_(seq):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((41, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array


def one_hot_encode_75(seq):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((75, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array


def one_hot_encode_100(seq):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((100, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array


def one_hot_encode_120(seq):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((120, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array


def one_hot_encode_90(seq):
    mapping = dict(zip("ACUG", range(4)))
    seq2 = [mapping[i] for i in seq]
    pre_padded = np.eye(4)[seq2]
    padded_array = np.zeros((90, 4))
    padded_array[0:len(seq), :] = pre_padded
    return padded_array