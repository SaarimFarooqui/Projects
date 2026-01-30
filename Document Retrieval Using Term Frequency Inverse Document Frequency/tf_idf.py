import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

document1 = "I have been playing video games since this morning."
document2 = "My favourite game has always been GTAV. It is an amazing game"
document3 = "Through online games, people can collectively solve large-scale computational problems."
document4 = "I am afraid gaming might affect my academic performace"

documents = [document1, document2, document3]

def tf_idf(documents):
  merged_documents = ""
  for document in documents:
    document = document.lower() + " "
    merged_documents += document 
  
  stop_words = set(stopwords.words('english')) #Storing stopwords in this variable
  
  list_of_words = [ word for word in word_tokenize(merged_documents) if word not in stop_words and word.isalnum() ]
  
  return list_of_words

result = tf_idf(documents)

print(result)