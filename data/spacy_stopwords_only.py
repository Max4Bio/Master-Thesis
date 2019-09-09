#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import os
import pickle
from train_model import add_vectors, evaluate, train_spacy_model_from_zero
from train_model import main, preprocess_annotations, fit_dataframe
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def show_model_stats(test_results, num_iter=100):
    test_dict = {"f-score": [], "precision": [], "recall": []}
    for tr in test_results:
        for score, value in tr.items():
            if score == "ents_f":
                test_dict["f-score"].append(value)
            elif score == "ents_p":
                test_dict["precision"].append(value)
            elif score == "ents_r":
                test_dict["recall"].append(value)
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
    # plt.show()
    plt.close()
    return test_df

# Hyperparameters
epochs = 30
dropout = 0.35
# Train and Test Data
TRAIN_DATA, TEST_DATA = pickle.load(open("train_test_data_for_all.pickle", 'rb'))


# In[4]:


# Read test document
with open("../../Daten/eval_docs/Doc_000.txt", 'r') as test_file:
    test_doc = test_file.read()


# In[5]:


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

    native_model_dir = "./models/de_core_news_md_stopwords_{}/".format(n)


    # ## Spacy de_core_news_md vectors

    # In[6]:


    # Train the NER of the de_core_news_md model
    test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                               model="de_core_news_md",
                                               validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[7]:

    native_df = show_model_stats(test_results=test_results)
    pickle.dump(native_df, open("data/dataframes/stats_of_de_core_news_md_stopwords_{}.pickle".format(n), 'wb'))
    pickle.dump(test_results, open("data/scorer/scorer_de_core_news_md_stopwords_{}.pickle".format(n), 'wb'))

    # ## vectors without german stopwords from nltk + de_core_news_md vectors

    # In[9]:


    word_vec_model = './models/stopwords_nltksw_plus_{}/'.format(n)
    glove_1 = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_nltksw.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    # wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_nltksw.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=glove_1,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[10]:

    del glove_1
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_stopwords_nltksw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_stopwords_nltksw_plus_{}.pickle".format(n), 'wb'))

    # ## vectors without german stopwords from nltk

    # In[12]:


    word_vec_model = './models/stopwords_nltksw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_nltksw.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[13]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_stopwords_nltksw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_stopwords_nltksw_alone_{}.pickle".format(n), 'wb'))

    # ## vectors without german stopwords from spacy + de_core_news_md vectors

    # In[15]:


    word_vec_model = './models/stopwords_spacysw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_spacysw.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[16]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_stopwords_spacysw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_stopwords_spacysw_plus_{}.pickle".format(n), 'wb'))

    # ## vectors without german stopwords from spacy

    # In[18]:


    word_vec_model = './models/stopwords_spacysw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_spacysw.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[19]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_stopwords_spacysw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_stopwords_spacysw_alone_{}.pickle".format(n), 'wb'))

    # ## vectors with german stopwords + de_core_news_md vectors

    # In[21]:


    word_vec_model = './models/stopwords_wsw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_wsw.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[22]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_stopwords_wsw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_stopwords_wsw_plus_{}.pickle".format(n), 'wb'))

    # ## vectors with german stopwords

    # In[24]:


    word_vec_model = './models/stopwords_wsw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_wsw.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[25]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_stopwords_wsw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_stopwords_wsw_alone_{}.pickle".format(n), 'wb'))

    # ## vectors without modified german stopwords from nltk + de_core_news_md vectors

    # In[27]:


    word_vec_model = './models/stopwords_modified_nltksw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_nltksw_modified.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[28]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_stopwords_modified_nltksw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_stopwords_modified_nltksw_plus_{}.pickle".format(n), 'wb'))

    # ## vectors without modified german stopwords from nltk

    # In[30]:


    word_vec_model = './models/stopwords_modified_nltksw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_nltksw_modified.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[31]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_stopwords_modified_nltksw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_stopwords_modified_nltksw_alone_{}.pickle".format(n), 'wb'))

    # ## vectors without modified german stopwords from spacy + de_core_news_md vectors

    # In[33]:


    word_vec_model = './models/stopwords_modified_spacysw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_spacysw_modified.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[34]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_stopwords_modified_spacysw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_stopwords_modified_spacysw_plus_{}.pickle".format(n), 'wb'))

    # ## vectors without modified german stopwords from spacy

    # In[36]:


    word_vec_model = './models/stopwords_modified_spacysw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_spacysw_modified.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[37]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_stopwords_modified_spacysw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_stopwords_modified_spacysw_alone_{}.pickle".format(n), 'wb'))

    # ## vectors with modified german stopwords + de_core_news_md vectors

    # In[39]:


    word_vec_model = './models/stopwords_modified_wsw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_wsw_modified.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[40]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_stopwords_modified_wsw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_stopwords_modified_wsw_plus_{}.pickle".format(n), 'wb'))

    # ## vectors with modified german stopwords

    # In[42]:


    word_vec_model = './models/stopwords_modified_wsw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/stopwords/word_vectors/spacy_preprocessed_wsw_modified.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[43]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_stopwords_modified_wsw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_stopwords_modified_wsw_alone_{}.pickle".format(n), 'wb'))

    # ## without vectors spacy model => Baseline

    # In[45]:


    word_vec_model = './models/lemma_baseline_{}/'.format(n)
    wo_model = spacy.load("de_core_news_md")
    wo_model.vocab.reset_vectors(width=300)
    wo_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wo_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[46]:

    del wo_model
    wo_df = show_model_stats(test_results=wo_test_results)
    pickle.dump(wo_df, open("data/dataframes/stats_of_stopwords_baseline_{}.pickle".format(n), 'wb'))
    pickle.dump(wo_test_results, open("data/scorer/scorer_stopwords_baseline_{}.pickle".format(n), 'wb'))

    # ## create pandas dataframe by concatenating all previous test dataframes

    # In[48]:


    # loading all previous dataframes
    native_df = pickle.load(open("data/dataframes/stats_of_de_core_news_md_stopwords_{}.pickle".format(n), 'rb'))
    stopwords_nltksw_plus = pickle.load(open("data/dataframes/stats_of_stopwords_nltksw_plus_{}.pickle".format(n), 'rb'))
    stopwords_nltksw_alone = pickle.load(open("data/dataframes/stats_of_stopwords_nltksw_alone_{}.pickle".format(n), 'rb'))
    stopwords_spacysw_plus = pickle.load(open("data/dataframes/stats_of_stopwords_spacysw_plus_{}.pickle".format(n), 'rb'))
    stopwords_spacysw_alone = pickle.load(open("data/dataframes/stats_of_stopwords_spacysw_alone_{}.pickle".format(n), 'rb'))
    stopwords_wsw_plus = pickle.load(open("data/dataframes/stats_of_stopwords_wsw_plus_{}.pickle".format(n), 'rb'))
    stopwords_wsw_alone = pickle.load(open("data/dataframes/stats_of_stopwords_wsw_alone_{}.pickle".format(n), 'rb'))
    stopwords_modified_nltksw_plus = pickle.load(open("data/dataframes/stats_of_stopwords_modified_nltksw_plus_{}.pickle".format(n), 'rb'))
    stopwords_modified_nltksw_alone = pickle.load(open("data/dataframes/stats_of_stopwords_modified_nltksw_alone_{}.pickle".format(n), 'rb'))
    stopwords_modified_spacysw_plus = pickle.load(open("data/dataframes/stats_of_stopwords_modified_spacysw_plus_{}.pickle".format(n), 'rb'))
    stopwords_modified_spacysw_alone = pickle.load(open("data/dataframes/stats_of_stopwords_modified_spacysw_alone_{}.pickle".format(n), 'rb'))
    stopwords_modified_wsw_plus = pickle.load(open("data/dataframes/stats_of_stopwords_modified_wsw_plus_{}.pickle".format(n), 'rb'))
    stopwords_modified_wsw_alone = pickle.load(open("data/dataframes/stats_of_stopwords_modified_wsw_alone_{}.pickle".format(n), 'rb'))
    baseline = pickle.load(open("data/dataframes/stats_of_stopwords_baseline_{}.pickle".format(n), 'rb'))


    # In[49]:


    comp_df = pd.concat([native_df.describe()["f-score"],
                         stopwords_nltksw_plus.describe()["f-score"],
                         stopwords_nltksw_alone.describe()["f-score"],
                         stopwords_spacysw_plus.describe()["f-score"],
                         stopwords_spacysw_alone.describe()["f-score"],
                         stopwords_wsw_plus.describe()["f-score"],
                         stopwords_wsw_alone.describe()["f-score"],
                         stopwords_modified_nltksw_plus.describe()["f-score"],
                         stopwords_modified_nltksw_alone.describe()["f-score"],
                         stopwords_modified_spacysw_plus.describe()["f-score"],
                         stopwords_modified_spacysw_alone.describe()["f-score"],
                         stopwords_modified_wsw_plus.describe()["f-score"],
                         stopwords_modified_wsw_alone.describe()["f-score"],
                         baseline.describe()["f-score"]
                        ], axis=1)


    merged = pd.concat([native_df["f-score"],
                         stopwords_nltksw_plus["f-score"],
                         stopwords_nltksw_alone["f-score"],
                         stopwords_spacysw_plus["f-score"],
                         stopwords_spacysw_alone["f-score"],
                         stopwords_wsw_plus["f-score"],
                         stopwords_wsw_alone["f-score"],
                         stopwords_modified_nltksw_plus["f-score"],
                         stopwords_modified_nltksw_alone["f-score"],
                         stopwords_modified_spacysw_plus["f-score"],
                         stopwords_modified_spacysw_alone["f-score"],
                         stopwords_modified_wsw_plus["f-score"],
                         stopwords_modified_wsw_alone["f-score"],
                         baseline["f-score"]
                        ], axis=1)

    dia_labels = [
        "de_core_news_md_{}".format(n),
        "stopwords_nltksw_plus_{}".format(n),
        "stopwords_nltksw_alone_{}".format(n),
        "stopwords_spacysw_plus_{}".format(n),
        "stopwords_spacysw_alone_{}".format(n),
        "stopwords_wsw_plus_{}".format(n),
        "stopwords_wsw_alone_{}".format(n),
        "stopwords_modified_nltksw_plus_{}".format(n),
        "stopwords_modified_nltksw_alone_{}".format(n),
        "stopwords_modified_spacysw_plus_{}".format(n),
        "stopwords_modified_spacysw_alone_{}".format(n),
        "stopwords_modified_wsw_plus_{}".format(n),
        "stopwords_modified_wsw_alone_{}".format(n),
        "baseline_{}".format(n)
    ]

    comp_df.columns = dia_labels
    merged.columns = dia_labels

    fig = plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=merged, dashes=False, palette=sns.color_palette("RdYlGn", n_colors=14))
    ax.set_title("Vorverarbeitung: Stoppwortentfernung, min_len=3 und ohne Anonymisierung", fontsize=22)
    ax.set_xlabel("Iteration des Trainingprozesses", fontsize=18)
    ax.set_ylabel("F1-Score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/stopwords_train_{}.png".format(n), format='png')
    # plt.show()
    plt.close()


    # In[53]:


    fig2 = plt.figure(figsize=(14, 8))
    ax2 = sns.lineplot(data=fit_dataframe(merged), dashes=False, palette=sns.color_palette("RdYlGn", n_colors=14))
    ax2.set_title("Vorverarbeitung: Stoppwortentfernung, min_len=3 und ohne Anonymisierung fitted", fontsize=22)
    ax2.set_xlabel("Iteration des Trainingprozesses", fontsize=18)
    ax2.set_ylabel("F1-Score in %", fontsize=18)
    ax2.set_ylim(0, 100)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/stopwords_fitted_train_{}.png".format(n), format='png')
    # plt.show()
    plt.close()


    # In[1]:


    bar_df = comp_df[comp_df.index == 'max']
    bar_df = bar_df.sort_values(by='max', axis=1)

    fig = plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=bar_df, palette=sns.color_palette("RdYlGn", n_colors=14))
    for i, cty in enumerate(bar_df.values.tolist()[0]):
        ax.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
    ax.set_title("Vorverarbeitung: Stoppwortentfernung, min_len=3 und ohne Anonymisierung Barplot", fontsize=22)
    ax.set_xlabel("Wortvektormodell", fontsize=18)
    ax.set_ylabel("Mittelwert F1-Score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/stopwords_barplot_{}.png".format(n), format='png')
    # plt.show()
    plt.close()
