#!/usr/bin/env python
# coding: utf-8

# Simple Recommender Engine using Weighted Average

# Task 1 - Library Import and File Unloading

# In[1]:


#Import Library dan File Unloading

#import library yang dibutuhkan
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

#lakukan pembacaan dataset
movie_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/title.basics.tsv', sep='\t') #untuk menyimpan title_basics.tsv
rating_df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/title.ratings.tsv', sep='\t') #untuk menyimpan title.ratings.tsv


# Task 2 - Cleaning table movie

# In[3]:


#5 data teratas dari tabel

print(movie_df.head())


# In[4]:


#Info data dari setiap kolom

print(movie_df.info())


# In[5]:


#Pengecekan Data dengan Nilai NULL

print(movie_df.isnull().sum())


# In[6]:


#Analisis Kolom dengan data bernilai NULL - part 1

print(movie_df.loc[(movie_df['primaryTitle'].isnull()) | (movie_df['originalTitle'].isnull())])


# In[7]:


#Membuang Data dengan Nilai NULL - part 1

#mengupdate movie_df dengan membuang data-data bernilai NULL
movie_df = movie_df.loc[(movie_df['primaryTitle'].notnull()) & (movie_df['originalTitle'].notnull())]

#menampilkan jumlah data setelah data dengan nilai NULL dibuang
print(len(movie_df))


# In[8]:


#Analisis Kolom dengan data bernilai NULL - part 2

movie_df = movie_df.loc[(movie_df['primaryTitle'].notnull()) & (movie_df['originalTitle'].notnull())]

print(movie_df.loc[movie_df['genres'].isnull()])


# In[9]:


#Membuang Data dengan Nilai NULL - part 2

#mengupdate movie_df dengan membuang data-data bernilai NULL
movie_df = movie_df.loc[movie_df['genres'].notnull()]

#menampilkan jumlah data setelah data dengan nilai NULL dibuang
print(len(movie_df))


# In[10]:


#Mengubah Nilai '\\N'

#mengubah nilai '\\N' pada startYear menjadi np.nan dan cast kolomnya menjadi float64
movie_df['startYear'] = movie_df['startYear']. replace('\\N',np.nan)
movie_df['startYear'] = movie_df['startYear']. astype('float64')
print(movie_df['startYear'].unique()[:5])

#mengubah nilai '\\N' pada endYear menjadi np.nan dan cast kolomnya menjadi float64
movie_df['endYear'] = movie_df['endYear']. replace('\\N',np.nan)
movie_df['endYear'] = movie_df['endYear']. astype('float64')
print(movie_df['endYear'].unique()[:5])

#mengubah nilai '\\N' pada runtimeMinutes menjadi np.nan dan cast kolomnya menjadi float64
movie_df['runtimeMinutes'] = movie_df['runtimeMinutes']. replace('\\N',np.nan)
movie_df['runtimeMinutes'] = movie_df['runtimeMinutes']. astype('float64')
print(movie_df['runtimeMinutes'].unique()[:5])


# In[11]:


#Mengubah nilai genres menjadi list

def transform_to_list(x):
    if ',' in x: 
    #ubah menjadi list apabila ada data pada kolom genre
        return x.split(',')
    else: 
    #jika tidak ada data, ubah menjadi list kosong
        return []

movie_df['genres'] = movie_df['genres'].apply(lambda x: transform_to_list(x))


# Task 3 - Cleaning table ratings

# In[12]:


#Menampilkan 5 data teratas

print(rating_df.head())


# In[13]:


#Menampilkan info data

print(rating_df.info())


# Task 4 - Joining table movie and table ratings

# In[14]:


#Inner Join table movie dan table rating

#Lakukan join pada kedua table
movie_rating_df = pd.merge(movie_df, rating_df, on='tconst', how='inner')

#Tampilkan 5 data teratas
print(movie_rating_df.head())

#Tampilkan tipe data dari tiap kolom
print(movie_rating_df.info())


# In[15]:


#Memperkecil ukuran Table

#Untuk memastikan bahwa sudah tidak ada lagi nilai NULL
print(movie_rating_df.info())


# Task 5 - Building Simple Recommender System

# In[16]:


#Pertanyaan 1: Berapa nilai C?

C = movie_rating_df['averageRating'].mean()
print(C)


# In[17]:


#Pertanyaan 2: Berapa nilai m?

m = movie_rating_df['numVotes'].quantile(0.8)
print(m)


# In[18]:


#Pertanyaan 3: Bagaimana cara membuat fungsi weighted formula?

def imdb_weighted_rating(df, var=0.8):
    v = df['numVotes']
    R = df['averageRating']
    C = df['averageRating'].mean()
    m = df['numVotes'].quantile(var)
    df['score'] = (v/(m+v))*R + (m/(m+v))*C #Rumus IMDb 
    return df['score']
    
imdb_weighted_rating(movie_rating_df)

#melakukan pengecekan dataframe
print(movie_rating_df.head())


# In[19]:


#Pertanyaan 4: Bagaimana cara membuat simple recommender system?

def simple_recommender(df, top=100):
    df = df.loc[df['numVotes'] >= m]
    df = df.sort_values(by='score', ascending=False) #urutkan dari nilai tertinggi ke terendah
    
    #Ambil data 100 teratas
    df = df[:top]
    return df
    
#Ambil data 25 teratas     
print(simple_recommender(movie_rating_df, top=25))


# In[20]:


#Pertanyaan 5: Bagaimana cara membuat simple recommender system dengan user preferences?

df = movie_rating_df.copy()

def user_prefer_recommender(df, ask_adult, ask_start_year, ask_genre, top=100):
    #ask_adult = yes/no
    if ask_adult.lower() == 'yes':
        df = df.loc[df['isAdult'] == 1]
    elif ask_adult.lower() == 'no':
        df = df.loc[df['isAdult'] == 0]

    #ask_start_year = numeric
    df = df.loc[df['startYear'] >= int(ask_start_year)]

    #ask_genre = 'all' atau yang lain
    if ask_genre.lower() == 'all':
        df = df
    else:
        def filter_genre(x):
            if ask_genre.lower() in str(x).lower():
                return True
            else:
                return False
        df = df.loc[df['genres'].apply(lambda x: filter_genre(x))]

    df = df.loc[df['numVotes'] >= m]  #Mengambil film dengan numVotes yang lebih besar atau sama dengan nilai m 
    df = df.sort_values(by='score', ascending=False)
    
    #jika kamu hanya ingin mengambil 100 teratas
    df = df[:top]
    return df

print(user_prefer_recommender(df,
                       ask_adult = 'no',
                        ask_start_year = 2000,
                       ask_genre = 'drama'
                       ))






