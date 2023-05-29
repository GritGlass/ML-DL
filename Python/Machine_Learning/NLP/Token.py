from nltk import sent_tokenize,word_tokenize
import nltk
#nltk.download('punkt')

#문장 토큰화
text_sample_KO='한글도 잘 되는가? \
테스트를 해보자 \
안되면 어쩔수 없고'

#단어 토큰
text_sample_EN='Hello, My name is Lily. Nice to meet you. I live in LA with my family.'

def tokenize_text(text):
    sentences = sent_tokenize(text=text)
    words = [word_tokenize(sentence) for sentence in sentences]
    return words

if __name__ == '__main__':
    words=tokenize_text(text_sample_EN)
    print(type(words),len(words))
    print(words)