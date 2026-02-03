from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import defaultdict

stop_words = set(stopwords.words('english')) 

#row_of_words returns a list of words in lower case which are alpha-numeric and not included in stop words.
def row_of_words(single_document):
  words = [word for word in word_tokenize(single_document.lower()) if word not in stop_words and word.isalnum()]
  return words

#The function table_of_words uses above function row_of_words to generate lists of  pre-processed words of each document. Each list contains preprocessed words from each document,  means no of lists = no of documents. These lists are the appended into another list which results into a lists within a list. Which serves a table.
def table_of_words(documents):
  table = []
  for document in documents:
    table.append(row_of_words(document))
  return table

#unique_words is a function that extracts unique words from table and returns a list
def unique_words(table):
  u_words = []
  for wordlist in table:
    for word in wordlist:
      if word not in u_words:
        u_words.append(word)
  return u_words
  
def term_frequency_table(list_documents, list_uniquewords, table_of_words):
  # The number of documents + 1 = number of columns.
  # The number of unique words + 1 = number of rows.
  frequency_table = [[0 for _ in range(len(list_documents)+1)] for _ in range(len(list_uniquewords)+1)]
  frequency_table[0][0] = "TERM"
  
  for index, word in enumerate(list_uniquewords):
    frequency_table[index+1][0] = word
    
  for index, word in enumerate(list_documents):
    frequency_table[0][index+1] = f"DOCUMENT {index+1}"
  
  for wordIndex, word in enumerate(list_uniquewords):
    for docIndex, row in enumerate(table_of_words):
      frequency_table[wordIndex+1][docIndex+1] = round(row.count(word) / len(row),3)
      
  return frequency_table

def document_frequency_table(table):
  doc_freq_dict = defaultdict(int)
  merged = []
  for row in table:
    merged += row
  for word in merged:
    doc_freq_dict[word] += 1
  return doc_freq_dict
  
  

document1 = "I have been playing video games since this morning."
document2 = "My favourite game has always been GTAV. It is an amazing game"
document3 = "Through online games, people can collectively solve large-scale computational problems."
document4 = "I am afraid gaming might affect my academic performace"
documents = [document1, document2, document3, document4]

table = table_of_words(documents)
uniquewords = unique_words(table)
freq_table = term_frequency_table(documents, uniquewords, table)

doc_freq = document_frequency_table(table)
for key, value in doc_freq.items():
  print(f"{key}: {value}")