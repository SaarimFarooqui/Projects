# TERM FREQUENCY-INVERSE DOCUMENT FREQUENCY

## About

***Term Frequency-Inverse Document Frequency (TF-IDF)*** is a statistical method and feature extraction technique. The core idea behind it is to find how important a word is in a document with respect to other documents. TF-IDF helps reduce the importance of common words inside a document, highlights words which are document specific, converts text into vectors and enables machine learning models to work on text.  

In short, it assigns higher weights to words that are frequent in a document but rare across the corpus. It is statistical and not semantic, does not consider word order and used in ***machine learning models*** like Linear Regression, Naive Bayes, and Support Vector Machine. 

## Description 
The code consists of multiple methods and the approach is to breakdown the problem into smaller tasks and perform each task using a method. The methods process the given input one by one and every single method is an essential part of the algorithm. This approach makes the over all algorithm easier to understand and detect logical errors in the calculation.  

Below is the necessary **explanation** of each method:  

### `1.  row_of_words()`  

This function **preprocesses the text** inside the document. It includes the documents of the corpus as well as the input document. The input taken is of string datatype and returns a list of only the words that are in **lowercase, alphanumeric, and non-stopwords**.

