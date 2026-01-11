import nltk
from nltk import sent_tokenize, word_tokenize

doc1 = "Octopus have three hearts and blue blood. Two hearts pump blood to the gills, while one pumps it to the rest of the body. Their blood is blue because it uses copper-based hemocyanin instead of iron."
doc2 = "Sloth spend almost their entire lives hanging upside down in trees. Their fur grows in the opposite direction (from stomach to back), allowing rainwater to flow off easily while they hang."
doc3 = "Camel can survive weeks without water. Their humps don’t store water—they store fat, which can be converted into energy and even small amounts of water when needed."
combine_string = doc1+" "+doc2+" "+doc3
words = word_tokenize(combine_string)
sentences = sent_tokenize(combine_string)
print(words)
print(sentences)