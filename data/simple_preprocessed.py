#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import pickle
import os
import numpy as np
from train_model import add_vectors, evaluate, main
from train_model import preprocess_annotations, fit_dataframe, train_spacy_model, train_spacy_model_from_zero
from auxillary_functions import load_sentences
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from spacy import displacy

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Hyperparmeters and global variables
epochs = 30
dropout = 0.35


# In[3]:


def show_model_stats(test_results, only_f1=False):
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
    ax.set_title("Statistics of de_core_news_md in {} iterations".format(num_iter), fontsize=22)
    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("Value of score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.close()
    # plt.show()
    return test_df

# Laod train and test data
TRAIN_DATA, TEST_DATA = pickle.load(open("train_test_data_for_all.pickle", 'rb'))


# In[5]:


# Read test document
with open("../../Daten/eval_docs/Doc_000.txt", 'r') as test_file:
    test_doc = test_file.read()


# In[6]:


LABELS = [
    "DrugName",
    "Strength",
    "Route",
    "Frequency",
    "Form",
    "Dose_Amount",
    "IntakeTime",
    "Duration",
    "Dispense_Amount",
    "Refill",
    "Necessity",
]

for n in range(10):

    native_model_dir = "./models/de_core_news_md_sp_{}/".format(n)

    # Train the NER of the de_core_news_md model
    test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                               model="de_core_news_md",
                                               validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[8]:

    native_df = show_model_stats(test_results=test_results, only_f1=False)
    pickle.dump(test_results, open("data/scorer/scorer_de_core_news_md_{}.pickle".format(n), 'wb'))
    pickle.dump(native_df, open("data/dataframes/stats_of_de_core_news_md_sp_{}.pickle".format(n), 'wb'))

    # In[10]:


    word_vec_model = './models/simple_prep_plus_{}/'.format(n)
    # glove_1 = spacy.load("de_core_news_md")
    # glove_1.vocab.vectors.from_glove("../../fastText/wordvectors/nltksw.vec")
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/simple_prep/word_vectors/simple_preprocessed.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  validation=TEST_DATA, model=wordvec_model,
                                                  n_iter=epochs, dropout=dropout)
    del wordvec_model


    # In[11]:


    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_test_results, open("data/scorer/scorer_simple_prep_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_df, open("data/dataframes/stats_of_simple_prep_plus_{}.pickle".format(n), 'wb'))

    # ## Only W2V in spacy model

    # In[13]:


    word_vec_model = './models/simple_prep_only_/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/simple_prep/word_vectors/simple_preprocessed.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    validation=TEST_DATA, model=only_model,
                                                    n_iter=epochs, dropout=dropout)
    del only_model


    # In[14]:


    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_test_results, open("data/scorer/scorer_simple_prep_only_{}.pickle".format(n), 'wb'))
    pickle.dump(only_df, open("data/dataframes/stats_of_simple_prep_only_{}.pickle".format(n), 'wb'))

    # ## without vectors spacy model

    # In[16]:


    word_vec_model = './models/simple_prep_baseline_{}/'.format(n)
    wo_model = spacy.load("de_core_news_md")
    wo_model.vocab.reset_vectors(width=300)
    wo_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  validation=TEST_DATA, model=wo_model,
                                                  n_iter=epochs, dropout=dropout)
    del wo_model


    # In[17]:


    wo_df = show_model_stats(test_results=wo_test_results)
    pickle.dump(wo_test_results, open("data/scorer/scorer_simple_prep_baseline_{}.pickle".format(n), 'wb'))
    pickle.dump(wo_df, open("data/dataframes/stats_of_simple_prep_baseline_{}.pickle".format(n), 'wb'))

    # In[18]:


    wo_df.describe()


    # In[19]:


    comp_df = pd.concat([native_df.describe()["f-score"], wv_df.describe()["f-score"],
                         wo_df.describe()["f-score"], only_df.describe()["f-score"]], axis=1)
    comp_df.columns = [
        "de_core_news_md_{}".format(n),
        "simple_prep_plus_{}".format(n),
        "simple_prep_only_{}".format(n),
        "baseline_{}".format(n)
    ]

    # In[21]:


    merged = pd.concat([native_df["f-score"], wv_df["f-score"],
                        only_df["f-score"], wo_df["f-score"]], axis=1)
    merged.columns = [
        "de_core_news_md_{}".format(n),
        "simple_prep_plus_{}".format(n),
        "simple_prep_only_{}".format(n),
        "baseline_{}".format(n)
    ]

    pickle.dump(merged, open('data/dataframes/f1_training_simple_preprocessed_{}.pickle'.format(n), 'wb'))


    # In[22]:


    merged = pickle.load(open('data/dataframes/f1_training_simple_preprocessed_{}.pickle'.format(n), 'rb'))

    # In[23]:


    fig = plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=merged)
    ax.set_title("Vorverarbeitung: Simple Preprocessed", fontsize=22)
    ax.set_xlabel("Iteration des Trainingprozesses", fontsize=18)
    ax.set_ylabel("F1-Score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/simple_preprocessed_train_{}.png".format(n), format='png')
    # plt.show()
    plt.close()


    # In[24]:


    fitted_df = pd.DataFrame(data=fit_dataframe(merged))


    # In[25]:


    fig2 = plt.figure(figsize=(14, 8))
    ax2 = sns.lineplot(data=fitted_df)
    ax2.set_title("Vorverarbeitung: Simple Preprocessed fitted", fontsize=22)
    ax2.set_xlabel("Iteration des Trainingprozesses", fontsize=18)
    ax2.set_ylabel("F1-Score in %", fontsize=18)
    ax2.set_ylim(0, 100)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/simple_preprocessed_fitted_train_{}.png".format(n), format='png')
    # plt.show()
    plt.close()

    # In[26]:


    comp_df = merged.describe()


    # In[27]:


    bar_df = comp_df[comp_df.index == 'max']
    bar_df = bar_df.sort_values(by='max', axis=1)

    fig = plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=bar_df, palette=sns.color_palette("RdYlGn", n_colors=4))
    for i, cty in enumerate(bar_df.values.tolist()[0]):
        ax.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
    ax.set_title("Vorverarbeitung: Simple Preprocessed Barplot", fontsize=22)
    ax.set_xlabel("Wortvektormodell", fontsize=18)
    ax.set_ylabel("Mittelwert F1-Score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/simple_preprocessed_barplot_{}.png".format(n), format='png')
    # plt.show()
    plt.close()
