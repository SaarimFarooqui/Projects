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
            TermFrequencyList.append(round(row.count(word)/len(row), 4))
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
    inversedocumentfrequency[key] = round(math.log(N/value), 4)
  
  return inversedocumentfrequency

def TF_IDF(termfrequency, inversedocumentfrequency):
  
  idfvalues = list(inversedocumentfrequency.values())
  tfidf = []
  tfidf.append(list(inversedocumentfrequency.keys()))
  for rowNum, row in enumerate(termfrequency):
    tfidfRow = []
    if rowNum != 0:
      for index, value in enumerate(row):
        tfidfRow.append(round(value * idfvalues[index],4))
      tfidf.append(tfidfRow)
    
  return tfidf

def InputVector(input, uniquewords, inversedocumentfrequency):
  vector = []
  inputWordFrequency = defaultdict(float)
  preprocessedRow = row_of_words(input)
  
  for word in uniquewords:
    inputWordFrequency[word] = preprocessedRow.count(word)/len(preprocessedRow)
  
  idfvalues = list(inversedocumentfrequency.values())
  inputWordFrequencies = list(inputWordFrequency.values())
  
  for index, value in enumerate(idfvalues):
    vector.append(round(idfvalues[index] * inputWordFrequencies[index],4))
  return vector
    
def DotProduct(vector1, vector2):
  DotProduct = 0
  for i in range(len(vector1)):
    DotProduct += vector1[i] * vector2[i]
  return round(DotProduct,4)

def Magnitude(vector):
  Magnitude= 0
  for value in vector:
    Magnitude += value ** 2
  return round(Magnitude ** 0.5, 4)

def CosineSimilarityRanking(inputvector, tfidf):
  numerator = 0
  denomerator = 0
  CosineSimilarityDictionary = defaultdict(float)
  tfidf.pop(0)
  for index, row in enumerate(tfidf):
    numerator = DotProduct(inputvector, row)
    denomerator = Magnitude(inputvector) * Magnitude(row)
    CosineSimilarityDictionary[f"Document: {index+1}"] = round (numerator / denomerator, 4)
    sorted_dict = dict(sorted(CosineSimilarityDictionary.items(), key=lambda x: x[1], reverse=True))
  return sorted_dict

document1 = "What is the future of Data Science"
document2 = "How can you use machine learning in personal life"
document3 = "Is this a good time to start learning data science and machine learning"
input = "Learning data science"
documents = [document1, document2, document3]

table = table_of_words(documents)
uniquewords = unique_words(table)
termfrequency = TermFrequencyLists(uniquewords, table)
documentfrequency =  DocumentFrequency(uniquewords, table)
inversedocumentfrequency = InverseDocumentFrequency(documentfrequency, documents)

tfidf = TF_IDF(termfrequency, inversedocumentfrequency)
inputvector = InputVector(input, uniquewords, inversedocumentfrequency)
cosinesimilarity = CosineSimilarityRanking(inputvector, tfidf)

tfidf = pd.DataFrame(tfidf)
print("TF IDF")
print(tfidf.to_string(index=False, header=False))
print("\n")
print("Input Vector")
print(inputvector)
print("\n")
for key, value in cosinesimilarity.items():
  print(f"{key} : {value}")