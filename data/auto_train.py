#!/usr/bin/env python

from subprocess import call


print("Executing simple_preprocessed.py ...")
call("python simple_preprocessed.py", shell=True)
print("Executing spacy_cistem.py ...")
call("python spacy_cistem.py", shell=True)
print("Executing spacy_lemma.py ...")
call("python spacy_lemma.py", shell=True)
print("Executing spacy_stopwords_only.py ...")
call("python spacy_stopwords_only.py", shell=True)
print("finished!")
