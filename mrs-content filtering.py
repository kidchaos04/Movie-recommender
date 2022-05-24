#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import ast


# In[4]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head()


# In[6]:


credits.head()


# In[7]:


movies=movies.merge(credits,on='title')


# movies.head()

# In[8]:


movies.head()


# In[9]:


movies.shape


# choosing columns
# 1.title
# 2.overview
# 3.geres
# 4.cast
# 5.crew
# 6.keywords

# In[10]:


movies=movies[['movie_id','title','genres','overview','keywords','cast','crew']]


# In[11]:


movies


# In[12]:


movies.isna().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.isna().sum()


# In[15]:


movies.duplicated().sum()


# In[16]:


movies['genres'][0]


# In[17]:


def convert_genre(obj):
    l=[]
    for i in ast.literal_eval(obj):
       l.append(i['name'])
    return l


# In[18]:


convert_genre('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[19]:


movies['genres']=movies['genres'].apply(convert_genre)


# In[20]:


movies.head()


# In[21]:


movies['keywords']=movies['keywords'].apply(convert_genre)


# In[22]:


movies.head()


# In[23]:


movies['cast']


# In[24]:


def convert_cast(obj):
    l=[]
    c=0
    for i in ast.literal_eval(obj):
        if c!=3:
            l.append(i['name'])
            c=c+1
        else:
            break
    return l        


# In[25]:


movies['cast']=movies['cast'].apply(convert_cast)


# In[26]:


movies.head()


# In[27]:


movies['crew'][0]


# In[28]:


def convert_crew(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l        


# In[29]:


movies['crew']=movies['crew'].apply(convert_crew)


# In[30]:


movies.head()


# In[31]:


movies['overview'][0]


# In[32]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[33]:


movies.head()


# In[34]:


movies['genres']=movies['genres'].apply(lambda x: [(i.replace(" ","")) for i in x])


# In[35]:


movies['genres']


# In[36]:


movies['cast']=movies['cast'].apply(lambda x: [(i.replace(" ","")) for i in x])
movies['crew']=movies['crew'].apply(lambda x: [(i.replace(" ","")) for i in x])


# In[37]:


movies['keywords']=movies['keywords'].apply(lambda x: [(i.replace(" ","")) for i in x])


# In[38]:


movies.head()


# In[39]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[40]:


movies


# In[41]:


new_df=movies[['movie_id','title','tags']]


# In[42]:


new_df


# In[43]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[44]:


new_df


# In[45]:


new_df['tags'][3]


# In[46]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[47]:


new_df


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[49]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[50]:


vectors


# In[51]:


cv.get_feature_names()


# In[52]:


import nltk


# In[53]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[54]:


def stemming(obj):
    y=[]
    for i in obj.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[55]:


new_df['tags']=new_df['tags'].apply(stemming)


# In[56]:


new_df['tags'][0]


# In[57]:


from sklearn.metrics.pairwise import cosine_similarity


# In[58]:


similarity=cosine_similarity(vectors)


# In[59]:


similarity[1]


# In[62]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[63]:


recommend('Avatar')


# In[64]:


recommend('Batman Begins')


# In[66]:


recommend('Spectre')


# In[67]:


import pickle


# In[68]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[69]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




