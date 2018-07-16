import spacy
from spacy import displacy

text = "Sebastian"

nlp = spacy.load('es_core_news_sm')



doc = nlp(text)
displacy.serve(doc, style='ent')

displacy.serve(doc, style='dep')
