import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

document1 = "I have been playing video games since this morning."
document2 = "My favourite game has always been GTAV. It is an amazing game"
document3 = "Through online games, people can collectively solve large-scale computational problems."
document4 = "I am afraid gaming might affect my academic performace"

documents = [document1, document2, document3]

#The function get_all_words() takes list of all documents, tokenizes all words, removes stop words and non-alpha numeric words and returns a list of those pre-processed words

def get_all_words(documents): 
  merged_documents = ""
  for document in documents:
    document = document.lower() + " "
    merged_documents += document 
  #stopwords stored in stop_words
  stop_words = set(stopwords.words('english')) 
  list_of_words = [ word for word in word_tokenize(merged_documents) if word not in stop_words and word.isalnum()]
  return list_of_words
  

def tf_idf(documents):
  l = get_all_words(documents)

result = tf_idf(documents)

print(result)