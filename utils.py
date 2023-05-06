"""Util functions used for each SSL method"""
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer
from unidecode import unidecode

tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")

def preprocess(x):
    """Preprocess input string x"""
    s = unidecode(x)
    s = str.lower(s)
    s = re.sub(r"\[[a-z]+\]","", s)
    s = re.sub(r"\*","", s)
    s = re.sub(r"[^a-zA-Z0-9]+"," ",s)
    s = re.sub(r" +"," ",s)
    s = re.sub(r"(.)\1+",r"\1",s)

    return s

def split_ssl_data(ids_array,mask_array,labels,num_classes,label_percent):
    """Split input data in labeled and unlabeled splits
    
    All input data is labeled. The unlabeled component is obtained
    by masking the corresponding labels
    
    """
    labeled = None
    unlabeled = None

    for class_idx in range(num_classes):
        class_ids = ids_array[labels==class_idx]
        class_mask = mask_array[labels==class_idx]
        sz = int(label_percent * class_ids.shape[0])

        labels_reduced = labels[labels==class_idx][:sz]
        labeled_ids, unlabeled_ids = class_ids[:sz], class_ids[sz:]
        labeled_mask, unlabeled_mask = class_mask[:sz], class_mask[sz:]

        if not labeled:
            labeled = (labeled_ids, labeled_mask, labels_reduced)
            unlabeled = (unlabeled_ids, unlabeled_mask)
        else:
            labeled = (
                np.concatenate([labeled[0],labeled_ids]),
                np.concatenate([labeled[1],labeled_mask]),
                np.concatenate([labeled[2],labels_reduced])
            )
            unlabeled = (
                np.concatenate([unlabeled[0],unlabeled_ids]),
                np.concatenate([unlabeled[1],unlabeled_mask]),
            )

    return labeled, unlabeled


def preprocess_robert(x):
    """tokenize input"""

    data = tokenizer(x,padding="max_length",max_length=96,truncation=True,return_tensors='np')
    return data["input_ids"], data["attention_mask"]

def map_func(input_ids, masks, labels):
    """map_func"""
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

def map_func2(input_ids, masks):
    """map_func2"""
    return {'input_ids': input_ids, 'attention_mask': masks}

def prepare_ds(filename,batch_size=64):
    """Prepare TF dataset from pandas dataframe"""

    df = pd.read_csv(filename)

    X_id_mask = df['text'].map(preprocess).apply(preprocess_robert).apply(pd.Series)

    X_id_mask.columns = ["input_ids","attention_mask"]

    ids_array = np.squeeze(np.stack(X_id_mask.input_ids.values), axis=1)
    mask_array = np.squeeze(np.stack(X_id_mask.attention_mask.values), axis=1)

    label_ids = {label_name:i for i, label_name in enumerate(sorted(set(df["label"])))}
    labels = df["label"].map(lambda x: label_ids[x]).values

    res_ds = tf.data.Dataset.from_tensor_slices((ids_array, mask_array, labels)).map(map_func).shuffle(len(df)).batch(batch_size)

    return res_ds


def prepare_train_ds(filename,batch_sizel=16, batch_sizeu=16,label_percent=0.05):
    """prepare training set with labeled and unlabeled splits"""

    df = pd.read_csv(filename)
    df = df.sample(frac=1)

    X_id_mask = df['text'].map(preprocess).apply(preprocess_robert).apply(pd.Series)

    X_id_mask.columns = ["input_ids","attention_mask"]

    ids_array = np.squeeze(np.stack(X_id_mask.input_ids.values), axis=1)
    mask_array = np.squeeze(np.stack(X_id_mask.attention_mask.values), axis=1)

    label_ids = {label_name:i for i, label_name in enumerate(sorted(set(df["label"])))}
    labels = df["label"].map(lambda x: label_ids[x]).values

    labeled, unlabeled = split_ssl_data(ids_array,mask_array,labels,len(label_ids),label_percent)
    labeled_ds = tf.data.Dataset.from_tensor_slices(labeled)
    labeled_ds = labeled_ds.map(map_func).shuffle(len(labeled_ds)).batch(batch_sizel).repeat()

    unlabeled_ds = tf.data.Dataset.from_tensor_slices(unlabeled)
    unlabeled_ds = unlabeled_ds.map(map_func2).shuffle(len(unlabeled_ds)).batch(batch_sizeu).repeat()

    return labeled_ds, unlabeled_ds
