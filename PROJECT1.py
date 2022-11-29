#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import pandas as pd


# In[2]:


nltk.download('vader_lexicon')


# In[17]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[18]:


sid = SentimentIntensityAnalyzer()


# In[19]:


# %%writefile reviewdata.txt
# WRITE YOUR TEXT HERE .... 


# In[20]:


with open('reviewdata.txt') as c:
    review = c.read()
    
df = pd.read_csv('reviewdata.txt', sep='\t')
df.head()


# In[21]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[22]:


doc = nlp(review[26:])


# In[23]:


for token in doc:
    print(f'{token.text:{15}} {token.pos_:{5}} {token.dep_:{10}} {token.lemma_:{15}} {spacy.explain(token.tag_)}')


# In[24]:


for ent in doc.ents:
    print(f"{ent.text:{10}}    {ent.label_:{10}}    {spacy.explain(ent.label_)}")


# In[25]:


from spacy import displacy


# In[26]:


displacy.render(doc, style='ent', jupyter=True)


# In[27]:


sid.polarity_scores(review)


# In[28]:


import colorama
from colorama import Fore


# In[29]:


def review_rating(string):
    scores = sid.polarity_scores(string)
    if scores['compound'] == 0:
        return 'Neutral'
    elif scores['compound'] > 0:
        return 'Positive'
    else:
        return 'Negative'


# In[30]:


if review_rating(review) == 'Neutral' :
    print(Fore.MAGENTA+'SENTIMENT ----> '+Fore.BLUE+'\033[1mNEUTRAL !!!\033[0m'+" \N{neutral face}")
elif review_rating(review) == 'Positive' :
    print(Fore.MAGENTA+'SENTIMENT ----> '+Fore.GREEN+'\033[1mPOSITIVE !!!\033[0m'+" \N{grinning face}")
elif review_rating(review) == 'Negative' :
    print(Fore.MAGENTA+'SENTIMENT ----> '+Fore.RED+'\033[1mNEGATIVE !!!\033[0m'+" \N{Angry Face}")


# In[ ]:





# In[ ]:




