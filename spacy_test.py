import spacy
from spacy import displacy
import dateparser

text= []

text.append("mañana a las 9pm, y el miercoles a las 14:00")

text.append("hoy a las 10pm")

text.append("el miercoles y el viernes")

text.append("el miercoles, el jueves y el domingo")

nlp = spacy.load('es_core_news_sm')


print(text)
for a in text:
	dates_found=[]
	print(a)
	for i in a.split(","):
		doc = nlp(i)
		cc = [token for token in doc if token.dep_ == "cc"]
		if len(cc) > 0:
			for x in cc:
				remove_strings = []
				for z in i.split(x.text):
					doc = nlp(z)
					remove_strings =  [token.text for token in doc if token.dep_ == "case" or token.dep_ == "det"]
					z = " ".join([h for h in z.split(" ") if h not in remove_strings])
					dates_found.append(dateparser.parse(z, languages=['es']))
				
		else:
			remove_strings =  [token.text for token in doc if token.dep_ == "case" or token.dep_ == "det"]
			i = " ".join([h for h in i.split(" ") if h not in remove_strings])
			dates_found.append(dateparser.parse(i, languages=['es']))
	
	print(dates_found)
