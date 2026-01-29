import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict

def summarize_text(text, summary_ratio = 0.3):
    sentences = sent_tokenize(text)
    
    words = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words("english"))
    
    filtered_words = [
        word for word in words if word.isalnum() and word not in stop_words
    ]
    
    word_freq = defaultdict(int)
    for word in filtered_words:
        word_freq[word] += 1
    
    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq
        
    sentence_scores = defaultdict(float)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] += word_freq[word]
    
    select_length = int(len(sentences) * summary_ratio)
    summary_sentences = sorted (
        sentence_scores,
        key = sentence_scores.get,
        reverse = True
    )[:select_length]
    
    return " ".join(summary_sentences)

if __name__ == "__main__":
    text = input("Enter the text to be summarized:\n\n")
    summary = summarize_text(text)
    print("\n--- Summary ---\n")
    print(summary)