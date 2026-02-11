from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import math
from collections import defaultdict

stop_words = set(stopwords.words('english')) 

#=======================================================METHOD==============================================================

#row_of_words returns a list of words in lower case which are alpha-numeric and not included in stop words.
def row_of_words(single_document):
  words = [word for word in word_tokenize(single_document.lower()) if word not in stop_words and word.isalnum()]
  return words

#=======================================================METHOD==============================================================

#The function table_of_words uses above function row_of_words to generate lists of  pre-processed words of each document. Each list contains preprocessed words from each document,  means no of lists = no of documents. These lists are the appended into another list which results into a lists within a list. Which serves a table.
def table_of_words(documents):
  table = []
  for document in documents:
    table.append(row_of_words(document))
  return table

#=======================================================METHOD==============================================================

#unique_words is a function that extracts unique words from table and returns a list
def unique_words(table):
  u_words = []
  for wordlist in table:
    for word in wordlist:
      if word not in u_words:
        u_words.append(word)
  return u_words

#=======================================================METHOD==============================================================
  
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

#=======================================================METHOD==============================================================

def document_frequency_table(table):
  doc_freq_dict = defaultdict(int)
  merged = []
  for row in table:
    merged += row
  for word in merged:
    doc_freq_dict[word] += 1
  return doc_freq_dict

#=======================================================METHOD==============================================================
# ⚠️ This method returns a dictionary (not a list or table) 

def inverse_document_frequency(documents ,doc_freq):
  inv_doc_freq = defaultdict(float)
  for key, value in doc_freq.items():
    inv_doc_freq[key] = round(math.log10(len(documents)/value),3)
  return inv_doc_freq

#=======================================================METHOD==============================================================
def tf_x_idf_vectorTable(documents, uniquewords, freq_table, inverse_doc_freq):
  rows = len(documents) + 1
  columns = len(uniquewords) + 1
  vector_table = [[0 for _ in range(columns)] for _ in range(rows)]
  
  for row_index, row in enumerate(vector_table):
    row[0] = f" DOCUMENT:{row_index} "
    
  vector_table[0][0] = " DOCs / TERMS "
  
  for index, word in enumerate(uniquewords):
    vector_table[0][index+1] = word
  
  inv_doc_freqs = list(inverse_doc_freq.values())
  
  for row_index, row in enumerate(vector_table):
    if row_index != 0:
      for idf_index, idf in enumerate(inv_doc_freqs):
        row[idf_index+1] = round(freq_table[idf_index+1][row_index] * idf, 3) 
  
  return vector_table
#=======================================================METHOD==============================================================

def tf_idf_tables_of_every_document(documents ,uniquewords, freq_table, inverse_doc_freq): 
  all_docs_tf_idf = []
  
  for doc_index, doc in enumerate(documents):
    table = [[0 for i in range(4)] for i in range(len(uniquewords)+1)]
    
    table[0][0:4] = "TERM", "TF", "IDF", "TF-IDF"
    
    #ADDING TERMS IN THE FIRST COLUMN
    for index, word in enumerate(uniquewords):
      table[index+1][0] = word
    
    #STORING THE RELATIVE DOCUMENT FREQUENCIES IN A LIST
    doc_frequency = [a[doc_index+1] for a in freq_table[1:]]
    
    #ADDING TERM FREQUENCIES IN THE SECOND COLUMN
    for index, word in enumerate(doc_frequency):
      table[index+1][1] = word
    
    #ADDING INVERSE DOCUMENT FREQUENCIES IN THE TABLE's COLUMN INDEX 2 (3rd column)
    keys_list = list(inverse_doc_freq.keys())
    for index, term in enumerate(keys_list):
      table[index+1][2] = inverse_doc_freq[term]
    
    for index, row in enumerate(table):
      if index != 0:
        row[3] = round(row[1] * row[2], 3)
      
    all_docs_tf_idf.append(table)

  return all_docs_tf_idf

#=======================================================METHOD==============================================================


#===========================================================================================================================

document1 = "AI models are trained on massive text datasets to understand and generate human language."
document2 = "Data science uses NLP to turn text into numbers, finding key themes in documents."
document3 = "Scientists analyze climate reports to track key terms like carbon emissions over time. Text analysis reveals trends in the research, strengthening evidence for human-driven change."

documents = [document1, document2, document3]

table = table_of_words(documents)
uniquewords = unique_words(table)
freq_table = term_frequency_table(documents, uniquewords, table)
doc_freq = document_frequency_table(table)
inverse_doc_freq_dictionary =  inverse_document_frequency(documents, doc_freq)

vector_table =  tf_x_idf_vectorTable(documents, uniquewords, freq_table, inverse_doc_freq_dictionary)

#df = pd.DataFrame(vector_table)
#df.to_html("VectorTable.html", index = False, header = False)

tf_idf_tables_list = tf_idf_tables_of_every_document(documents, uniquewords, freq_table, inverse_doc_freq_dictionary)

for table_number, table in enumerate(tf_idf_tables_list):
  print(f"_______DOCUMENT_NUMBER_{table_number + 1}_______"),
  df = pd.DataFrame(table)
  print(df.to_string(index=False, header=False))
  print("\n")