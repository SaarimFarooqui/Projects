from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english')) 

# row_of_words returns a list of words in lower case which are alpha-numeric and not included in stop words.
def row_of_words(single_document):
  words = [word.lower() for word in word_tokenize(single_document) if word not in stop_words and word.isalnum()]
  return words

def table_of_words(documents):
  table_of_elements = []
  for document in documents:
    table_of_elements.append(row_of_words(document))
  return table_of_elements

# unique_words is a function that extracts unique words from table and returns a list
def unique_words(table):
  u_words = []
  for wordlist in table:
    for word in wordlist:
      if word not in u_words:
        u_words.append(word)
  return u_words

document1 = "I have been playing video games since this morning."
document2 = "My favourite game has always been GTAV. It is an amazing game"
document3 = "Through online games, people can collectively solve large-scale computational problems."
document4 = "I am afraid gaming might affect my academic performace"
documents = [document1, document2, document3, document4]

table = table_of_words(documents)

extracted = unique_words(table)

empty_table = [[0 for _ in range(len(documents))] for _ in range(len(extracted))]

for i in empty_table:
  print(i)