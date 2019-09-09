#!/usr/bin/env python
# coding: utf8
from pathlib import Path
import os
import spacy
import pickle
import os
import numpy as np
from train_model import add_vectors, evaluate, main
from train_model import preprocess_annotations, fit_dataframe, train_spacy_model
from sklearn.model_selection import train_test_split
from spacy import displacy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_annotations(ann_file):
    with open(ann_file, 'r') as annf:
        ann_content = [ar.strip().split("\t") for ar in annf.readlines()]
    annotations = [ac[1].split() for ac in ann_content]
    annotations = [(int(a[1]), int(a[2]), a[0]) for a in annotations]
    return annotations

def make_spacy_annotations(txt_file, ann_file, whodoc=False):
    with open(txt_file, 'r') as txt_reader:
        txt_content = txt_reader.read()
    annotations = read_annotations(ann_file)
    if whodoc and not annotations:
        return False
    annot_txt = (txt_content, {"entities": annotations})
    return annot_txt

def load_narrative(train_dir="/home/hoody/brat-nightly-2018-12-12/data/kd_freitexte/", whodoc=False):
    ann_files = sorted([train_dir + ld for ld in os.listdir(train_dir) if ld.endswith(".ann")])
    txt_files = sorted([train_dir + ld for ld in os.listdir(train_dir) if ld.endswith(".txt")])
    annotations = []
    for txt, ann in zip(txt_files, ann_files):
        annotations.append(make_spacy_annotations(txt, ann, whodoc=whodoc))
    annotations = [a for a in annotations if a]
    X = [ann[0] for ann in annotations]
    y = [ann[1] for ann in annotations]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    TRAIN_DATA = list(zip(X_train, y_train))
    TEST_DATA = list(zip(X_test, y_test))
    TEST_DATA = [(td[0], td[1]["entities"]) for td in TEST_DATA]
    return TRAIN_DATA, TEST_DATA

def load_sentences():
    annotations = preprocess_annotations("raw/files.txt")
    X = [ann[0] for ann in annotations]
    y = [ann[1] for ann in annotations]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    TRAIN_DATA = list(zip(X_train, y_train))
    TEST_DATA = list(zip(X_test, y_test))
    TEST_DATA = [(td[0], td[1]["entities"]) for td in TEST_DATA]
    return TRAIN_DATA, TEST_DATA

def load_wholedoc():
    return load_narrative(train_dir="/home/hoody/brat-nightly-2018-12-12/data/Medikation_AB/", whodoc=True)

def show_model_stats(test_results, only_f1=False, n=3):
    if only_f1:
        test_dict = {"f-score": [], "loss": []}
    else:
        test_dict = {"f-score": [], "precision": [], "recall": [], "loss": []}
    for tr in test_results:
        for score, value in tr.items():
            if score == "ents_f":
                test_dict["f-score"].append(value)
            elif score == "ents_loss":
                test_dict["loss"].append(value)
            if not only_f1:
                if score == "ents_p":
                    test_dict["precision"].append(value)
                elif score == "ents_r":
                    test_dict["recall"].append(value)
    test_dict["loss"] = list((np.array(test_dict["loss"]) / max(test_dict["loss"])) * 100)              
    num_iter = len(test_dict["f-score"])
    # print(test_dict)
    test_df = pd.DataFrame(data=test_dict)
    fig = plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=test_df)
    ax.set_title("Statistics of de_core_news_md in {0} iterations (n={1})".format(num_iter, n), fontsize=22)
    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("Value of score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()
    return test_df

def plott_training_df(test_df, n=3):
    num_iter = len(test_df)
    fig = plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=test_df)
    ax.set_title("Statistics of de_core_news_md in {0} iterations (n={1})".format(num_iter, n), fontsize=22)
    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("Value of score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.show()

def calc_mean_df(list_of_dfs):
    merged_results = pd.concat(list_of_dfs).groupby(level=0).mean()
    return merged_results


if __name__ == '__main__':
    pass