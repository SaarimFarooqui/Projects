from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import math
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
  
def inverse_document_frequency(documents ,doc_freq):
  inv_doc_freq = defaultdict(float)
  for key, value in doc_freq.items():
    inv_doc_freq[key] = round(math.log10(len(documents)/value),3)
  return inv_doc_freq

def tf_idf(documents ,uniquewords, freq_table, inverse_doc_freq): 
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
    
document1 = "Recent advances in artificial intelligence, particularly in large language models, have revolutionized how machines process and generate human-like text. These models are trained on vast datasets encompassing books, articles, and websites, allowing them to identify patterns and relationships between words and concepts. The core challenge lies in developing algorithms that can understand context and nuance, moving beyond simple keyword matching. This progress raises important questions about the future of work, creativity, and the ethical implications of increasingly capable AI systems."

document2 = "In the field of data science, the ability to extract meaningful insights from unstructured text data is a critical skill. Techniques such as natural language processing (NLP) are employed to categorize documents, gauge sentiment, and identify prevalent themes. The fundamental step in this process often involves converting text into a numerical format, where the importance of specific words within a document and across a corpus can be quantified. This quantitative analysis allows researchers and businesses to make data-driven decisions based on trends found in customer feedback, news articles, or technical reports."

document3 = "The analysis of historical climate documents and scientific papers is crucial for modeling future environmental changes. Researchers systematically process decades of reports to identify key terms and trends related to temperature rise, carbon emissions, and ecosystem degradation. By applying statistical models to this corpus of text, scientists can track the frequency and evolution of critical concepts over time. This textual evidence, combined with numerical data, strengthens the consensus on human-driven climate change and informs international policy debates."

document4 = "Modern medical research increasingly relies on computational tools to sift through millions of academic journals and patient records. Text mining algorithms help discover connections between symptoms, genes, and treatments by analyzing the language used in scientific literature. Identifying the most significant terms and phrases across a collection of clinical studies can accelerate drug discovery and highlight emerging health threats. This approach enables a more efficient synthesis of global knowledge, ultimately aiming to improve diagnostic accuracy and patient outcomes through evidence-based medicine."

documents = [document1, document2, document3, document4]

table = table_of_words(documents)
uniquewords = unique_words(table)
freq_table = term_frequency_table(documents, uniquewords, table)
doc_freq = document_frequency_table(table)
inverse_doc_freq =  inverse_document_frequency(documents, doc_freq)
termF_invDF = tf_idf(documents, uniquewords, freq_table, inverse_doc_freq)

for table_number, table in enumerate(termF_invDF):
  print(f"_______DOCUMENT_NUMBER_{table_number + 1}_______"),
  df = pd.DataFrame(table)
  print(df.to_string(index=False, header=False))
  print("\n")
