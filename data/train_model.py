#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pre-trained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy import displacy
import os
import re
import numpy as np
import pandas as pd
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from scipy.optimize import curve_fit
from tqdm import tqdm_notebook


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(train_data, model=None, new_model_name="medical", output_dir=None, n_iter=30, 
         test_doc="Das ist ein Test.", labels=["MED"], display=False, print_loss=True, test_model=True, test_eval=None):

    """Set up the pipeline and entity recognizer, and train the new entity."""
    LABELS = labels
    TRAIN_DATA = train_data
    if model is not None:
        if type(model) != str:
            nlp = model
        else:
            nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("de")  # create blank Language class
        print("Created blank 'de' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    nlp.remove_pipe("ner")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)
    
    [ner.add_label(LABEL) for LABEL in LABELS]  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    # if model is None:
    optimizer = nlp.begin_training()
    print("begin training...")
    # else:
        # optimizer = nlp.resume_training()
        # print("resume training...")
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(4.0, 32.0, 1.001)
        # batch up the examples using spaCy's minibatch
        test_result = []
        # random.seed(101)
        for itn in tqdm_notebook(range(n_iter)):
            # print("Epoch {} of {}".format(itn, n_iter))
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            if print_loss:
                print("Losses", losses)
            if test_eval:
                test_result.append(evaluate(ner_model=nlp, test_data=test_eval, print_result=False))
    
    # test the trained model
    test_text = test_doc
    doc = nlp(test_text)    

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        if test_model:
            # test the saved model
            print("Loading from", output_dir)
            nlp2 = spacy.load(output_dir)
            # Check the classes have loaded back consistently
            assert nlp2.get_pipe("ner").move_names == move_names
            doc2 = nlp2(test_text)
            if display:
                displacy.render(doc2, style='ent', jupyter=True)
            del nlp2
    if test_eval:
        return test_result
    del nlp

def preprocess_annotations(annot_dir):
    if os.path.isdir(annot_dir):
        ann_files = [os.path.join(annot_dir, ld) for ld in sorted(list(os.listdir(annot_dir))) if ld.endswith(".ann")]
        txt_files = [os.path.join(annot_dir, ld) for ld in sorted(list(os.listdir(annot_dir))) if ld.endswith(".txt")]
    elif os.path.isfile(annot_dir):
        # insert the brat dir of your annotationfiles
        files_path = "/home/hoody/brat-nightly-2018-12-12/data/ft_extracts/" #<- here
        print("Annotation files path: ", files_path)
        with open(annot_dir, 'r') as ad:
            ad_lines = ad.readlines()
            ad_content = [al.strip().split()[-1] for al in ad_lines if al.strip()]
        
        ann_files = [os.path.join(files_path, ld) for ld in ad_content if ld.endswith(".ann")]
        txt_files = [os.path.join(files_path, ld) for ld in ad_content if ld.endswith(".txt")]
    else:
        print("You have to enter a directory or file path!")
    train_data = []
    for af, tf in zip(ann_files, txt_files):
        with open(tf, 'r') as tf_reader:
            tf_content = tf_reader.read()
        with open(af, 'r') as af_reader:
            af_content = [ar.strip().split("\t") for ar in af_reader.readlines() if ar.startswith("T")]
        annotations = [(ac[1].split()[0], int(ac[1].split()[1]), int(ac[1].split()[2]), ac[2]) for ac in af_content]
        new_ann = {}
        # print(annotations)
        for sent in re.finditer(r'.+', tf_content):
            for label, start, end, name in annotations:
                sent_start, sent_end = sent.span()
                if sent_start <= start < sent_end and sent_start < end <= sent_end:
                    feat_text = sent.group()
                    feat_hit = re.search(name, feat_text)
                    if feat_hit:
                        fh_start, fh_end = feat_hit.span()
                        if feat_text not in new_ann.keys():
                            new_ann[feat_text] = {"entities": []}
                        if label == "NAN":
                            continue
                        else:
                            new_ann[feat_text]["entities"].append((fh_start, fh_end, label))
        train_data += list(new_ann.items())
    return train_data

def evaluate(ner_model, test_data, print_result=True, loss=None):
    scorer = Scorer()
    for input_, annot in test_data:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    if print_result:
        for score_name, f_score in scorer.scores.items():
            if score_name == "ents_f": 
                print(f"{score_name}:     {f_score}")
            elif score_name == "ents_p":
                print(f"{score_name}:     {f_score}")
            elif score_name == "ents_r":
                print(f"{score_name}:     {f_score}")
    score_dict = scorer.scores
    score_dict["ents_loss"] = loss
    return score_dict

'''
def add_vectors(pipe, vector_path):
    # Add word vectors in txt-format to spacy nlp 
    with open(vector_path, 'r') as vectors:
        vectors_txt = vectors.readlines()
    vectors_dict = {}
    for vt in vectors_txt:
        line = vt.split()
        np_vec = np.array(line[1:], dtype=float)
        vectors_dict[vt[0]] = np.pad(np_vec, (0, 300 - np_vec.shape[0]), 'constant', constant_values=0.0)
    for word, vector in vectors_dict.items():
        pipe.vocab.set_vector(word, vector)
    return pipe
'''

def add_vectors(vectors_loc, lang=None, pipe=False):
    if lang is None:
        nlp = Language()
    elif pipe:
        nlp = pipe
    else:
        # create empty language class – this is required if you're planning to
        # save the model to disk and load it back later (models always need a
        # "lang" setting). Use 'xx' for blank multi-language class.
        nlp = spacy.blank(lang)
    with open(vectors_loc, "rb") as file_:
        header = file_.readline()
        nr_row, nr_dim = header.split()
        # nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in file_:
            line = line.rstrip().decode("utf8")
            pieces = line.rsplit(" ", int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype="f")
            try:
                nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
            except ValueError:
                continue
    # test the vectors and similarity
    text = "Tumor Operation"
    doc = nlp(text)
    print(text, doc[0].similarity(doc[1]))
    return nlp

def fit_dataframe(df):
    epochs = len(df)
    def f(x, a, b, n):
        return a * x ** n  / (x ** n + b)

    fitted_curves = {}
    for c in df.columns:
        values = df[c].values.tolist()
        x = list(range(1, epochs + 1)) 
        popt, pcov = curve_fit(f, x, values, p0=[1800., 20., 1.])
        fitted_curves[c + " fitted"] = f(x, *popt)
    fitted_df = pd.DataFrame(data=fitted_curves)
    return fitted_df

def train_spacy_model(train_data, labels, model=None, new_model_name="medication", validation=None,
                      output_dir=None, n_iter=30, test_model=False, dropout=0.35):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    TRAIN_DATA = train_data
    if model is not None:
        if type(model) != str:
            nlp = model
            print("Existing model '%s'" % model)
        else:
            nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        print("No NER in pipe found: creating new NER")
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        print("Taking existing NER in pipe")
        ner = nlp.get_pipe("ner")

    [ner.add_label(LABEL) for LABEL in labels] # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        test_result = []
        for itn in tqdm_notebook(range(n_iter)):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
            print("Epoch: {}, Losses: {}".format(itn + 1, losses), end='\r')
            # test actual train performance
            if validation:
                test_result.append(evaluate(ner_model=nlp, loss=losses['ner'], test_data=validation, print_result=False))

    # test the trained model
    test_text = "Möchten Sie eine antibiotische Abschirmung intravenös mit Cefuroxim 50 über 2 Wochen als Infusion?"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
        
        if test_model:
            # test the saved model
            print("Loading from", output_dir)
            nlp2 = spacy.load(output_dir)
            # Check the classes have loaded back consistently
            assert nlp2.get_pipe("ner").move_names == move_names
            doc2 = nlp2(test_text)
            for ent in doc2.ents:
                print(ent.label_, ent.text)
    del nlp
    return test_result

def train_spacy_model_from_zero(train_data, labels, model=None, new_model_name="medication", validation=None,
                      output_dir=None, n_iter=30, test_model=False, dropout=0.35):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    TRAIN_DATA = train_data
    if model is not None:
        if type(model) != str:
            nlp = model
            print("Existing model '%s'" % model)
        else:
            nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        print("No NER in pipe found: creating new NER")
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        print("Taking existing NER in pipe")
        ner = nlp.get_pipe("ner")
        
    [ner.add_label(LABEL) for LABEL in labels] # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    # if model is None:
    optimizer = nlp.begin_training()
    # else:
        # optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(4.0, 32.0, 1.001)
        # batch up the examples using spaCy's minibatch
        test_result = []
        last_run = 0.0
        for itn in tqdm_notebook(range(n_iter)):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
            print("Epoch: {}, Losses: {}".format(itn + 1, losses), end='\r')
            # test actual train performance
            if validation:
                evaluation = evaluate(ner_model=nlp, loss=losses['ner'], test_data=validation, print_result=False)
                test_result.append(evaluation)
                actual_f = evaluation["ents_f"]
                last_run = actual_f                

    # test the trained model
    test_text = "Möchten Sie eine antibiotische Abschirmung intravenös mit Cefuroxim 50 über 2 Wochen als Infusion? "\
                "Intraoperativ wurde zudem eine intravenöse Antibiose mit Cefuroxim und Clont eingeleitet."
    doc = nlp(test_text)
    print("Entities in: '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)
        
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model with f_score of {} to".format(last_run), output_dir)

    del nlp
    return test_result