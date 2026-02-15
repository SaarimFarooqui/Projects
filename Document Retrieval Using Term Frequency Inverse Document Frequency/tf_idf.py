from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import math
from collections import defaultdict
stop_words = set(stopwords.words('english')) 

# Returns a list of preprocessed words out of a single string
def row_of_words(single_document):
  words = [word for word in word_tokenize(single_document.lower()) if word not in stop_words and word.isalnum()]
  return words

# Returns a list of lists of preprocessed words
def table_of_words(documents):
  table = []
  for document in documents:
    table.append(row_of_words(document))
  return table

# Returns unique words found in a list of lists aka nested lists
def unique_words(table):
  u_words = []
  for wordlist in table:
    for word in wordlist:
      if word not in u_words:
        u_words.append(word)
  return u_words


def TermFrequencyLists(uniquewords, table):
    TermFrequencyLists = []
    TermFrequencyLists.append(uniquewords)
    
    for row in table:
        TermFrequencyList = []
        for word in uniquewords:
            TermFrequencyList.append(round(row.count(word)/len(row), 3))
        TermFrequencyLists.append(TermFrequencyList)
    
    return TermFrequencyLists

def DocumentFrequency(uniquewords, table):
  DocumentFrequency = defaultdict(int)
  for i in uniquewords:
    for doc in table:
      if i in doc:
        DocumentFrequency[i] += 1
  return DocumentFrequency
  
def InverseDocumentFrequency(documentfrequency, documents):
  inversedocumentfrequency = defaultdict(float)
  N = len(documents)
  
  for key, value in documentfrequency.items():
    inversedocumentfrequency[key] = round(math.log(N/value), 3)
  
  return inversedocumentfrequency

def TF_IDF(termfrequency, inversedocumentfrequency):
  
  idfvalues = list(inversedocumentfrequency.values())
  tfidf = []
  tfidf.append(list(inversedocumentfrequency.keys()))
  for rowNum, row in enumerate(termfrequency):
    tfidfRow = []
    if rowNum != 0:
      for index, value in enumerate(row):
        tfidfRow.append(round(value * idfvalues[index],3))
      tfidf.append(tfidfRow)
    
  return tfidf

def InputVector(input, uniquewords, inversedocumentfrequency):
  InputVectorwithHeadings = []
  InputVectorwithHeadings.append(uniquewords)
  vector = []
  inputWordFrequency = defaultdict(float)
  preprocessedRow = row_of_words(input)
  
  for word in uniquewords:
    inputWordFrequency[word] = preprocessedRow.count(word)/len(preprocessedRow)
  
  idfvalues = list(inversedocumentfrequency.values())
  inputWordFrequencies = list(inputWordFrequency.values())
  
  for index, value in enumerate(idfvalues):
    vector.append(round(idfvalues[index] * inputWordFrequencies[index],3))
  InputVectorwithHeadings.append(vector)
  
  return InputVectorwithHeadings
    
  
document1 = "data science machine"
document2 = "data science"
document3 = "machine learning"
documents = [document1, document2, document3]

table = table_of_words(documents)
uniquewords = unique_words(table)
termfrequency = TermFrequencyLists(uniquewords, table)
documentfrequency =  DocumentFrequency(uniquewords, table)
inversedocumentfrequency = InverseDocumentFrequency(documentfrequency, documents)
tfidf = TF_IDF(termfrequency, inversedocumentfrequency)

df = pd.DataFrame(tfidf)
df.to_html("tfldf.html", index = False, header = False)