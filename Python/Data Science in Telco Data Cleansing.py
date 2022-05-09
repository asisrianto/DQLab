#!/usr/bin/env python
# coding: utf-8

# Library dan Data yang Digunakan

# Library yang Digunakan
# 
# Pada analisis kali ini, akan digunakan beberapa package yang membantu kita dalam melakukan analisis data.
# 
#     Pandas
# 
#     Pandas (Python for Data Analysis) adalah library Python yang fokus untuk proses analisis data seperti manipulasi data, persiapan data, dan pembersihan data.
#         read_csv() digunakan untuk membaca file csv
#         str.match() digunakan untuk mencocokan dengan karakter tertentu
#         drop() digunakan untuk menghapus
#         count() digunakan untuk menghitung masing-masing variable
#         drop_duplicates() digunakan untuk menghapus data duplicate rows
#         fillna() digunakan untuk mengisi dengan nilai tertentu
#         quantile() digunakan untuk melihat quantile ke tertentu
#         mask() mengganti nilai tertentu jika kondisi memenuhi
#         astype() mengubah tipe data
#         value_counts() digunakan untuk menghitung unik dari kolom
#         sort_values() digunakan untuk sort values
#         isnull() digunakan untuk mendeteksi missing values
#         dropna() digunakan untuk menghapus missing values
#         replace() digunakan untuk mengganti nilai
# 
#     Matplotlib
# 
#     Matplotlib adalah library Python yang fokus pada visualisasi data seperti membuat plot grafik. Matplotlib dapat digunakan dalam skrip Python, Python dan IPython shell, server aplikasi web, dan beberapa toolkit graphical user interface (GUI) lainnya.
#         figure() digunakan untuk membuat figure gambar baru
# 
#     Seaborn
# 
#     Seaborn membangun di atas Matplotlib dan memperkenalkan tipe plot tambahan. Ini juga membuat plot Matplotlib tradisional Anda terlihat sedikit lebih cantik.
#         box_plot() digunakan untuk membuat box plot
# 

# Data yang Digunakan
# 
# Untuk dataset yang digunakan sudah disediakan dalam format csv, silahkan baca melalui fungsi pandas di python df_load = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqlab_telco.csv')
#  
# 
# Untuk detail datanya adalah sebagai berikut:
# 
#     UpdatedAt Periode of Data taken
#     customerID Customer ID
#     gender Whether the customer is a male or a female (Male, Female)
#     SeniorCitizen Whether the customer is a senior citizen or not (1, 0)
#     Partner Whether the customer has a partner or not (Yes, No)
#     Dependents Whether the customer has dependents or not (Yes, No)
#     tenure Number of months the customer has stayed with the company
#     PhoneService Whether the customer has a phone service or not (Yes, No)
#     MultipleLines Whether the customer has multiple lines or not (Yes, No, No phone service)
#     InternetService Customer’s internet service provider (DSL, Fiber optic, No)
#     OnlineSecurity Whether the customer has online security or not (Yes, No, No internet service)
#     OnlineBackup Whether the customer has online backup or not (Yes, No, No internet service)
#     DeviceProtection Whether the customer has device protection or not (Yes, No, No internet service)
#     TechSupport Whether the customer has tech support or not (Yes, No, No internet service)
#     StreamingTV Whether the customer has streaming TV or not (Yes, No, No internet service)
#     StreamingMovies Whether the customer has streaming movies or not (Yes, No, No internet service)
#     Contract The contract term of the customer (Month-to-month, One year, Two year)
#     PaperlessBilling Whether the customer has paperless billing or not (Yes, No)
#     PaymentMethod The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
#     MonthlyCharges The amount charged to the customer monthly
#     TotalCharges The total amount charged to the customer
#     Churn Whether the customer churned or not (Yes or No)

# In[1]:


#import library
import pandas as pd
pd.options.display.max_columns = 50

#import dataset
df_load = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/dqlab_telco.csv')

#Tampilkan jumlah baris dan kolom
print(df_load.shape)

#Tampilkan 5 data teratas
print(df_load.head(5))

#Jumlah ID yang unik
print(df_load.customerID.nunique())


# Mencari Validitas ID Number Pelanggan

# In[2]:


#Memfilter ID Number Pelanggan Format Tertentu

df_load['valid_id'] = df_load['customerID'].astype(str).str.match(r'(45\d{9,10})')
df_load = (df_load[df_load['valid_id'] == True]).drop('valid_id', axis = 1)
print('Hasil jumlah ID Customer yang terfilter adalah', df_load['customerID'].count())


# In[3]:


#Memfilter Duplikasi ID Number Pelanggan

# Drop Duplicate Rows
df_load.drop_duplicates()
# Drop duplicate ID sorted by Periode
df_load = df_load.sort_values('UpdatedAt', ascending=False).drop_duplicates('customerID')
print('Hasil jumlah ID Customer yang sudah dihilangkan duplikasinya (distinct) adalah',df_load['customerID'].count())


# Mengatasi Missing Values

# In[4]:


#Mengatasi Missing Values dengan Penghapusan Rows

print('Total missing values data dari kolom Churn',df_load['Churn'].isnull().sum())
# Dropping all Rows with spesific column (churn)
df_load.dropna(subset=['Churn'], inplace=True)
print('Total Rows dan kolom Data setelah dihapus data Missing Values adalah',df_load.shape)


# In[5]:


#Mengatasi Missing Values dengan Pengisian Nilai tertentu

print('Status Missing Values :',df_load.isnull().values.any())
print('\nJumlah Missing Values masing-masing kolom, adalah:')
print(df_load.isnull().sum().sort_values(ascending=False))

#handling missing values Tenure fill with 11
df_load['tenure'].fillna(11, inplace=True)

#Loop
#Handling missing values num vars (except Tenure)
for col_name in list(['MonthlyCharges','TotalCharges']):
    #write your command here
	median = df_load[col_name].median()
	df_load[col_name].fillna(median, inplace=True)
	
print('\nJumlah Missing Values setelah di imputer datanya, adalah:')
print(df_load.isnull().sum().sort_values(ascending=False))


# Mengatasi Outlier

# In[6]:


#Mendeteksi adanya Outlier (Boxplot)

print('\nPersebaran data sebelum ditangani Outlier: ')
print(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())

# Creating Box Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Misal untuk kolom tenure
plt.figure()
sns.boxplot(x=df_load['tenure'])
plt.show()
# dan seterusnya untuk kedua kolom yang tersisa secara berurut
plt.figure()
sns.boxplot(x=df_load['MonthlyCharges'])
plt.show()
plt.figure()
sns.boxplot(x=df_load['TotalCharges'])
plt.show()


# In[7]:


#Mengatasi Outlier

# Handling with IQR
Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)

IQR = Q3 - Q1
maximum  = Q3 + (1.5*IQR)
print('Nilai Maximum dari masing-masing Variable adalah: ')
print(maximum)
minimum = Q1 - (1.5*IQR)
print('\nNilai Minimum dari masing-masing Variable adalah: ')
print(minimum)

more_than     = (df_load > maximum)
lower_than    = (df_load < minimum)
df_load       = df_load.mask(more_than, maximum, axis=1) 
df_load       = df_load.mask(lower_than, minimum, axis=1)

print('\nPersebaran data setelah ditangani Outlier: ')
print(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())


# Menstandarisasi Nilai

# In[8]:


#Mendeteksi Nilai yang tidak Standar

df_load['tenure'].fillna(11, inplace=True)
for col_name in list(['MonthlyCharges','TotalCharges']):
    median = df_load[col_name].median()
    df_load[col_name].fillna(median, inplace=True)

Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)
IQR = Q3 - Q1
maximum  = Q3 + (1.5*IQR)
minimum = Q1 - (1.5*IQR)

more_than     = (df_load > maximum)
lower_than    = (df_load < minimum)
df_load       = df_load.mask(more_than, maximum, axis=1) 
df_load       = df_load.mask(lower_than, minimum, axis=1)

# Masukkan variable
for col_name in list(['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']):
    print('\nUnique Values Count \033[1m' + 'Before Standardized \033[0m Variable',col_name)
    print(df_load[col_name].value_counts())


# In[9]:


#Menstandarisasi Variable Kategorik

Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)
Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)
IQR = Q3 - Q1
maximum  = Q3 + (1.5*IQR)
minimum = Q1 - (1.5*IQR)

more_than     = (df_load > maximum)
lower_than    = (df_load < minimum)
df_load       = df_load.mask(more_than, maximum, axis=1) 
df_load       = df_load.mask(lower_than, minimum, axis=1)

df_load = df_load.replace(['Wanita','Laki-Laki','Churn','Iya'],['Female','Male','Yes','Yes'])

# Masukkan variable
for col_name in list(['gender','Dependents','Churn']):
    print('\nUnique Values Count \033[1m' + 'After Standardized \033[0mVariable',col_name)
    print(df_load[col_name].value_counts())

