# coding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np
import pprint

print(tf.__version__)


def get_filenames_by_prefix(source_dir, prefix_name):
    all_files = os.listdir(source_dir)
    results = []
    for filename in all_files:
        if filename.startswith(prefix_name):
            results.append(os.path.join(source_dir, filename))
    return results


source_dir = "./DNN_tfrecords/"
train_filenames = get_filenames_by_prefix(source_dir, "train")

pprint.pprint(train_filenames)

expected_features = {
    "ctrl_feature": tf.io.VarLenFeature(dtype=tf.string),
    "cnop_feature": tf.io.VarLenFeature(dtype=tf.string)}


def parse_example(serialize_example):
    example = tf.io.parse_single_example(serialize_example, expected_features)

    ctrl_feature = tf.sparse.to_dense(example["ctrl_feature"], default_value=b"")
    cnop_feature = tf.sparse.to_dense(example["cnop_feature"], default_value=b"")

    ctrl_feature = tf.io.decode_raw(ctrl_feature, tf.float32)
    cnop_feature = tf.io.decode_raw(cnop_feature, tf.float32)
    return ctrl_feature, cnop_feature


def tfrecords_to_dataset(filenames, nreads=5, batch_size=200, n_parse_threads=5,
                         shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type="GZIP"),
        cycle_length=nreads)
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_example, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


tfrecords_train = tfrecords_to_dataset(train_filenames)




