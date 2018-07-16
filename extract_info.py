import re
import nltk
import spacy
from nltk.corpus import stopwords

nlp = spacy.load('es_core_news_sm')
stop = stopwords.words('english')


def extract_phone_numbers(string):
    r = re.compile(r'(\d{4}[-\.\s]??\d{3}[-\.\s]??\d{3}|\(\d{4}\)\s*\d{3}[-\.\s]??\d{3}|\d{3}[-\.\s]??\d{3}[-\.\s]??\d{3}|\d{3}[-\.\s]??\d{3}|\+\d{3}[-\.\s]??\d{3}[-\.\s]??\d{3}[-\.\s]??\d{3})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    doc = nlp(document)
    ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
    for e in ents:
        if e[3] == 'PER':
            names.append(e[0])

    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    a = ' '.join([c[0] for c in chunk])
                    if a not in names:
                        names.append(a)
    return names

print(extract_phone_numbers("mi numero es 0981 585 772"))
print(extract_phone_numbers("mi numero es 0981-585-772"))
print(extract_phone_numbers("mi numero es +595981585772"))
print(extract_names("mi nombre es Sebastian Rojas y mi hermano es Joel"))

