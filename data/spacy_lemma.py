#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import os
import pickle
from train_model import add_vectors, evaluate, main, train_spacy_model_from_zero
from train_model import preprocess_annotations, fit_dataframe
from sklearn.model_selection import train_test_split

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


# In[4]:


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

    native_model_dir = "./models/de_core_news_md_lemma_{}/".format(n)


    # ## Spacy de_core_news_md vectors

    # In[7]:


    # Train the NER of the de_core_news_md model
    test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                               model="de_core_news_md",
                                               validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[8]:


    native_df = show_model_stats(test_results=test_results)
    pickle.dump(native_df, open("data/dataframes/stats_of_de_core_news_md_lemma_{}.pickle".format(n), 'wb'))
    pickle.dump(test_results, open("data/scorer/scorer_de_core_news_md_lemma_{}.pickle".format(n), 'wb'))


    # ## spacy lemmas vectors without german stopwords from nltk + de_core_news_md vectors

    # In[10]:


    word_vec_model = './models/slemmas_nltksw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_nltksw_spacy_lemma_2.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[11]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_spacy_lemma_nltksw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_spacy_lemma_nltksw_plus_{}.pickle".format(n), 'wb'))

    # ## spacy lemmas vectors without german stopwords from nltk

    # In[13]:


    word_vec_model = './models/slemmas_nltksw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_nltksw_spacy_lemma_2.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[14]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_spacy_lemma_nltksw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_spacy_lemma_nltksw_alone_{}.pickle".format(n), 'wb'))

    # ## spacy lemmas vectors without german stopwords from spacy + de_core_news_md vectors

    # In[16]:


    word_vec_model = './models/slemmas_spacysw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wgsw_spacy_lemma_2.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[17]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_spacy_lemma_spacysw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_spacy_lemma_spacysw_plus_{}.pickle".format(n), 'wb'))

    # ## spacy lemmas vectors without german stopwords from spacy

    # In[19]:


    word_vec_model = './models/slemmas_spacysw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wgsw_spacy_lemma_2.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)

    # In[20]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_spacy_lemma_spacysw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_spacy_lemma_spacysw_alone_{}.pickle".format(n), 'wb'))

    # ## spacy lemmas vectors with german stopwords + de_core_news_md vectors

    # In[22]:


    word_vec_model = './models/slemmas_wsw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wsw_spacy_lemma_2.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)
    # In[23]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_spacy_lemma_wsw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_spacy_lemma_wsw_plus_{}.pickle".format(n), 'wb'))
    # ## spacy lemmas vectors with german stopwords

    # In[25]:


    word_vec_model = './models/slemmas_wsw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wsw_spacy_lemma_2.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[26]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_spacy_lemma_wsw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_spacy_lemma_wsw_alone_{}.pickle".format(n), 'wb'))

    # ## germalemmas vectors without german stopwords from nltk + de_core_news_md vectors

    # In[28]:


    word_vec_model = './models/germalemma_nltksw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_nltksw_germalemma_2.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[29]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_germalemma_nltksw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_germalemma_nltksw_plus_{}.pickle".format(n), 'wb'))

    # ## germalemmas vectors without german stopwords from nltk

    # In[31]:


    word_vec_model = './models/germalemma_nltksw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_nltksw_germalemma_2.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[32]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_germalemma_nltksw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_germalemma_nltksw_alone_{}.pickle".format(n), 'wb'))

    # ## germalemmas vectors without german stopwords from spacy + de_core_news_md vectors

    # In[34]:


    word_vec_model = './models/germalemma_spacysw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wgsw_germalemma_2.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[35]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_germalemma_spacysw_plus_{}.pickle".format(n), 'wb'))
    pickle.dump(wv_test_results, open("data/scorer/scorer_germalemma_spacysw_plus_{}.pickle".format(n), 'wb'))

    # ## germalemmas vectors without german stopwords from spacy

    # In[37]:


    word_vec_model = './models/germalemma_spacysw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wgsw_germalemma_2.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[38]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_germalemma_spacysw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_germalemma_spacysw_alone_{}.pickle".format(n), 'wb'))

    # ## germalemmas vectors with german stopwords + de_core_news_md vectors

    # In[40]:


    word_vec_model = './models/germalemma_wsw_plus_{}/'.format(n)
    wordvec_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wsw_germalemma_2.txt', lang=1, pipe=spacy.load("de_core_news_md"))
    wv_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wordvec_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[41]:

    del wordvec_model
    wv_df = show_model_stats(test_results=wv_test_results)
    pickle.dump(wv_df, open("data/dataframes/stats_of_germalemma_wsw_plus_{}.pickle".format(n), 'wb'))

    # ## germalemmas vectors with german stopwords

    # In[43]:


    word_vec_model = './models/germalemma_wsw_alone_{}/'.format(n)
    only_model = spacy.load("de_core_news_md")
    only_model.vocab.reset_vectors(width=300)
    only_model = add_vectors('../../Word2Vec-Versuche/preprocessing/lemma/word_vectors/spacy_preprocessed_wsw_spacy_lemma_2.txt', lang=1, pipe=only_model)
    only_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                    model=only_model,
                                                    validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[44]:

    del only_model
    only_df = show_model_stats(test_results=only_test_results)
    pickle.dump(only_df, open("data/dataframes/stats_of_germalemma_wsw_alone_{}.pickle".format(n), 'wb'))
    pickle.dump(only_test_results, open("data/scorer/scorer_germalemma_wsw_alone_{}.pickle".format(n), 'wb'))

    # ## without vectors spacy model => Baseline

    # In[46]:


    word_vec_model = './models/lemma_baseline_{}/'.format(n)
    wo_model = spacy.load("de_core_news_md")
    wo_model.vocab.reset_vectors(width=300)
    wo_test_results = train_spacy_model_from_zero(train_data=TRAIN_DATA, labels=LABELS,
                                                  model=wo_model,
                                                  validation=TEST_DATA, n_iter=epochs, dropout=dropout)


    # In[47]:

    del wo_model
    wo_df = show_model_stats(test_results=wo_test_results)
    pickle.dump(wo_df, open("data/dataframes/stats_of_lemma_baseline_{}.pickle".format(n), 'wb'))
    pickle.dump(wo_test_results, open("data/scorer/scorer_lemma_baseline_{}.pickle".format(n), 'wb'))


    # In[48]:


    wo_df.describe()


    # ## create pandas dataframe by concatenating all previous test dataframes

    # In[2]:


    # loading all previous dataframes
    native_df = pickle.load(open("data/dataframes/stats_of_de_core_news_md_lemma_{}.pickle".format(n), 'rb'))
    slemma_nltksw_plus = pickle.load(open("data/dataframes/stats_of_spacy_lemma_nltksw_plus_{}.pickle".format(n), 'rb'))
    slemma_nltksw_alone = pickle.load(open("data/dataframes/stats_of_spacy_lemma_nltksw_alone_{}.pickle".format(n), 'rb'))
    slemma_spacysw_plus = pickle.load(open("data/dataframes/stats_of_spacy_lemma_spacysw_plus_{}.pickle".format(n), 'rb'))
    slemma_spacysw_alone = pickle.load(open("data/dataframes/stats_of_spacy_lemma_spacysw_alone_{}.pickle".format(n), 'rb'))
    slemma_wsw_plus = pickle.load(open("data/dataframes/stats_of_spacy_lemma_wsw_plus_{}.pickle".format(n), 'rb'))
    slemma_wsw_alone = pickle.load(open("data/dataframes/stats_of_spacy_lemma_wsw_alone_{}.pickle".format(n), 'rb'))
    germalemma_nltksw_plus = pickle.load(open("data/dataframes/stats_of_germalemma_nltksw_plus_{}.pickle".format(n), 'rb'))
    germalemma_nltksw_alone = pickle.load(open("data/dataframes/stats_of_germalemma_nltksw_alone_{}.pickle".format(n), 'rb'))
    germalemma_spacysw_plus = pickle.load(open("data/dataframes/stats_of_germalemma_spacysw_plus_{}.pickle".format(n), 'rb'))
    germalemma_spacysw_alone = pickle.load(open("data/dataframes/stats_of_germalemma_spacysw_alone_{}.pickle".format(n), 'rb'))
    germalemma_wsw_plus = pickle.load(open("data/dataframes/stats_of_germalemma_wsw_plus_{}.pickle".format(n), 'rb'))
    germalemma_wsw_alone = pickle.load(open("data/dataframes/stats_of_germalemma_wsw_alone_{}.pickle".format(n), 'rb'))
    baseline = pickle.load(open("data/dataframes/stats_of_lemma_baseline_{}.pickle".format(n), 'rb'))


    # In[3]:


    comp_df = pd.concat([native_df.describe()["f-score"],
                         slemma_nltksw_plus.describe()["f-score"],
                         slemma_nltksw_alone.describe()["f-score"],
                         slemma_spacysw_plus.describe()["f-score"],
                         slemma_spacysw_alone.describe()["f-score"],
                         slemma_wsw_plus.describe()["f-score"],
                         slemma_wsw_alone.describe()["f-score"],
                         germalemma_nltksw_plus.describe()["f-score"],
                         germalemma_nltksw_alone.describe()["f-score"],
                         germalemma_spacysw_plus.describe()["f-score"],
                         germalemma_spacysw_alone.describe()["f-score"],
                         germalemma_wsw_plus.describe()["f-score"],
                         germalemma_wsw_alone.describe()["f-score"],
                         baseline.describe()["f-score"]
                        ], axis=1)


    merged = pd.concat([native_df["f-score"],
                         slemma_nltksw_plus["f-score"],
                         slemma_nltksw_alone["f-score"],
                         slemma_spacysw_plus["f-score"],
                         slemma_spacysw_alone["f-score"],
                         slemma_wsw_plus["f-score"],
                         slemma_wsw_alone["f-score"],
                         germalemma_nltksw_plus["f-score"],
                         germalemma_nltksw_alone["f-score"],
                         germalemma_spacysw_plus["f-score"],
                         germalemma_spacysw_alone["f-score"],
                         germalemma_wsw_plus["f-score"],
                         germalemma_wsw_alone["f-score"],
                         baseline["f-score"]
                        ], axis=1)

    dia_labels = [
        "de_core_news_md_{}".format(n),
        "spacy_lemma_nltksw_plus_{}".format(n),
        "spacy_lemma_nltksw_alone_{}".format(n),
        "spacy_lemma_spacysw_plus_{}".format(n),
        "spacy_lemma_spacysw_alone_{}".format(n),
        "spacy_lemma_wsw_plus_{}".format(n),
        "spacy_lemma_wsw_alone_{}".format(n),
        "germalemma_nltksw_plus_{}".format(n),
        "germalemma_nltksw_alone_{}".format(n),
        "germalemma_spacysw_plus_{}".format(n),
        "germalemma_spacysw_alone_{}".format(n),
        "germalemma_wsw_plus_{}".format(n),
        "germalemma_wsw_alone_{}".format(n),
        "baseline_{}".format(n)
    ]

    comp_df.columns = dia_labels
    merged.columns = dia_labels

    fig = plt.figure(figsize=(14, 8))
    ax = sns.lineplot(data=merged, dashes=False, palette=sns.color_palette("RdYlGn", n_colors=14))
    ax.set_title("Vorverarbeitung: Lemmatisierung", fontsize=22)
    ax.set_xlabel("Iteration des Trainingprozesses", fontsize=18)
    ax.set_ylabel("F1-Score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/lemma_train_{}.png".format(n), format='png')
    # plt.show()
    plt.close()


    # In[7]:


    fig2 = plt.figure(figsize=(14, 8))
    ax2 = sns.lineplot(data=fit_dataframe(merged), dashes=False, palette=sns.color_palette("RdYlGn", n_colors=14))
    ax2.set_title("Vorverarbeitung: Lemmatisierung fitted", fontsize=22)
    ax2.set_xlabel("Iteration des Trainingprozesses", fontsize=18)
    ax2.set_ylabel("F1-Score in %", fontsize=18)
    ax2.set_ylim(0, 100)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/lemma_fitted_train_{}.png".format(n), format='png')
    # plt.show()
    plt.close()


    # In[8]:


    bar_df = comp_df[comp_df.index == 'max']
    bar_df = bar_df.sort_values(by='max', axis=1)

    fig = plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=bar_df, palette=sns.color_palette("RdYlGn", n_colors=14))
    for i, cty in enumerate(bar_df.values.tolist()[0]):
        ax.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
    ax.set_title("Vorverarbeitung: Lemmatisierung Barplot", fontsize=22)
    ax.set_xlabel("Wortvektormodell", fontsize=18)
    ax.set_ylabel("Mittelwert F1-Score in %", fontsize=18)
    ax.set_ylim(0, 100)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)
    plt.savefig("data/figures/lemma_barplot_{}.png".format(n), format='png')
    # plt.show()
    plt.close()

    # pickle.dump((TRAIN_DATA, TEST_DATA), open("train_test_data_for_all.pickle", 'wb'))
