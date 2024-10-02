#LIBRARIES
import io
import random
import string 
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 


#READING THE CORPUS

with open('coi.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOKENIZATION

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

#PREPROCESSING

lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


#KEY WORD MATCHING 

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am happy that You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# RESPONSE GENERATION

def response(user_response):
    chatbot_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        chatbot_response=chatbot_response+"I am sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response = chatbot_response+sent_tokens[idx]
        return chatbot_response


flag=True
print("CHATBOT: My name is chatting bot . I will answer your questions  about Constitution of india . If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("CHATBOT: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("CHATBOT: "+greeting(user_response))
            else:
                print("CHATBOT: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("CHATBOT : Bye! take care..")    
        
        

