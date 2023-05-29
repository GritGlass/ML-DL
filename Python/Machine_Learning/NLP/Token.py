from nltk import sent_tokenize,word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import nltk
nltk.download('wordnet')

#문장 토큰화
text_sample_KO='한글도 잘 되는가? \
테스트를 해보자 \
안되면 어쩔수 없고'

text_sample_EN='Hello, My name is Lily. Nice to meet you. I live in LA with my family.'

def tokenize_text(text):
    sentences = sent_tokenize(text=text)
    words = [word_tokenize(sentence) for sentence in sentences]
    return words

def remove_stopwords(words):
    stopwords=nltk.corpus.stopwords.words('english')
    all_tokens=[]

    for s in words:
        filtered = []
        for w in s:
            word=w.lower()
            if word not in stopwords:
                filtered.append(w)
        all_tokens.append(filtered)
    return all_tokens

def stemming(word):
    stemmer=LancasterStemmer()
    stem=stemmer.stem(word)
    return stem

def lemmatization(word,pos):
    lammatizer=WordNetLemmatizer()
    lamma=lammatizer.lemmatize(word,pos)
    return lamma

if __name__ == '__main__':
    text=text_sample_EN
    words=tokenize_text(text)
    filtered_tokens=remove_stopwords(words)
    print(filtered_tokens)

    stem_ex=['killed','killing','kills']
    pos='v'
    for ss in stem_ex:
        stem=stemming(ss)
        lamma=lemmatization(ss, pos)

        print(lamma)
