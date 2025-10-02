#!/usr/bin/env python
# coding: utf-8

# In[1]:


# (1) Latar Belakang
# Perusahaan transportasi taksi memiliki tantangan dalam menjaga produktivitas driver dan kepuasan pelaggan dengan tetap menjaga optimasi pendapatan
# Berdasarkan data operasional, kami melihat adanya fenomena yang perlu ditinjau lebih dalam seperti ditemukannya trip dengan jarak 0, variasi pemasukan per jam, perbedaan demand pada jam tertentu, serta lokasi pickup yang mendominasi permintaan customer
# Analisis ini penting untuk mengoptimalkan productivity driver, meningkatkan efisiensi alokasi armada, generate revenue, serta menjaga loyalitas customer


# In[ ]:


# (2) Pernyataan Masalah 
# - Masih ditemukan perjalanan dengan jarak tempuh nol yang terjadi secara berulang. Perlu ditinjau lebih dalam apakah case ini signifikan dan diperlukan monitoring lebih lanjut, mengingat hal tersebut berpotensi dengan fraud, pembatalan, dan operasional yang tidak efisien
# - Jam shift driver yang belum merata. Jam pagi relatif rendah, sementara menumpuk di jam siang hingga sore. Bagaimana melakukan pengaturan shift agar tetap balance dan menghasilkan revenue yang optimal?
# - Terdapat demand tinggi pada zona pickup tertentu, namun distribusi driver masih belum merata. Sehingga perlu dilakukan analisis demand berdasarkan lokasi agar perusahaan dapat melakukan penempatan armada yang tepat sasaran


# In[2]:


# (3) Data
# Berikut adalah analisis data yang dilakukan untuk menjawab permasalahan diatas 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import normaltest, chi2_contingency, mannwhitneyu, ttest_ind


# In[3]:


# (3.1) Detail dari dataset terdapat 20 kolom, yaitu sebagai berikut : 

# VendorID
# lpep_pickup_datetime
# lpep_dropoff_datetime
# store_and_fwd_flag
# RatecodeID
# PULocationID
# DOLocationID
# passenger_count
# trip_distance	
# fare_amount
# extra
# mta_tax
# tip_amount
# tolls_amount	
# ehail_fee
# improvement_surcharge
# total_amount
# payment_type
# trip_type
# congestion_surcharge


# In[4]:


# (3.2) Berikut 5 baris teratas dan terbawah dari dataset NYC TLC Trip Record

df=pd.read_csv(r"C:\Users\T490s\OneDrive\Desktop\PURWADHIKA COURSE\NYC TLC Trip Record.csv")
display(df.head(),df.tail())


# In[5]:


# (4) Data Understanding and Cleaning
# (4.1) Berikut adalah informasi kolom dan jenis data dari dataset

print(f'Jumlah baris dan kolom di dataset NYC TLC Trip Record adalah {df.shape}')
df.info()


# In[6]:


# (4.2) Cek statistik deskriptif dari dataset
# Berdasarkan statistik deskriptif, didapatkan informasi sebagai berikut

display(df.describe(), df.describe(include='object'))


# In[7]:


# (4.3) Cek missing value dari dataset
# Kami menemukan data null ada di kolom berikut dengan detail jumlahnya 
# ehail_fee, trip_type, congestion_surcharge, store_and_fwd_flag, RatecodeID, payment_type, dan passenger_count 

df.isna().sum().sort_values(ascending=False)


# In[8]:


# (5) Identifikasi & Cleaning Data
# (5.1) Apabila kita amati, kolom ehail_fee null dengan jumlah 68211 dimana data yang terhitung ada sebanyak 68211. Sehingga dapat disimpulakn semua data ehail_fee kosong, jadi kita akan hapus dari dataset

df_cleaned = df.copy()
df_cleaned = df.drop(columns=['ehail_fee'], errors='ignore')
df_cleaned


# In[9]:


# (5.2) Jumlah data dari nilai trip_distance == 0
# Berdasarkan statistik deskriptif, didapatkan bahwa nilai min dari kolom trip_distance adalah 0 yang rtinya tidak ada perjalanan, sehingga data ini akan kita pisahkan untuk dicari tahu rootcause-nya
# Ditemukan persentase data trip_distance dengan nilai 0 ada sebanyak 4.9%. Sehingga ada sekitar 3350 trip dengan distance 0, jadi jumlah ini cukup signifikan untuk dicari tahu rootcause-nya

df_cleaned['zero_trip_distance'] = (df_cleaned['trip_distance'] == 0)
df_cleaned['zero_trip_distance'].value_counts()


# In[10]:


zero_trips = df_cleaned[df_cleaned['zero_trip_distance']]
print("Jumlah trip_distance = 0 sebanyak", len(zero_trips))
print("Persentase trip_distance = 0 adalah", len(zero_trips) / len(df_cleaned) * 100)


# In[11]:


# (5.3) Filter dataset fare_amount dan total_amount nilainya minus
# Ditemukan lagi dari statistik deskriptif bahwa nilai min dari fare_amount ada yang minus -70.000000 dan -71.500000
# Namun, ketika dilakukan pengecekan persentasenya 0.3 persen. Sehingga data ini dapat kita abaikan karena tidak signifikan 

# Persentase fare_amount dengan nilai <= 0 sebanyak 259 dengan persentase 0.3
df_cleaned['minus_fare_amount'] = (df_cleaned['fare_amount'] <= 0)
df_cleaned['minus_fare_amount'].value_counts()


# In[13]:


print(df_cleaned['minus_fare_amount'].mean() * 100)


# In[14]:


# Persentase total_amount dengan nilai <= 0 sebanyak 248 dengan persentase 0.3
df_cleaned['minus_total_amount'] = (df_cleaned['total_amount'] <= 0)
df_cleaned['minus_total_amount'].value_counts()


# In[15]:


print(df_cleaned['minus_total_amount'].mean() * 100)


# In[16]:


# Filter yang nilainya > 0. Hasilnya sama dengan diatas dimana 
# - minus_fare_amount (False) = 67952
# - minus_total_amount (False) = 67963

df_cleaned = df_cleaned[df_cleaned['fare_amount']>0]
df_cleaned = df_cleaned[df_cleaned['total_amount']>0]


# In[17]:


# (5.4) Cek duplikat dari dataset
# Ditemukan bahwa data tidak ada yang duplikat

df_cleaned.duplicated().sum()


# In[18]:


# (5.5) Cek kembali data yang sudah dilakukan cleaning
# Kolom e_hail fee sudah tidak ada 
# Kolom fare_amount dengan nilai <= 0 sudah tidak ada
# Kolom total_amount dengan nilai <= 0 sudah tidak ada

print(f'Jumlah baris dan kolom di dataset NYC TLC Trip Record adalah {df_cleaned.shape}')
df_cleaned.info()


# In[19]:


# Jumlah data berkurang karena adanya overlap baris
overlap = df[(df['fare_amount'] <= 0) & (df['total_amount'] <= 0)]
print("Jumlah overlap:", len(overlap))


# In[20]:


df_cleaned.to_csv("NYC_TLC_cleaned.csv", index=False)


# In[27]:


# (6) Data Analysis
# (6.1) Menguji signifikansi trip_distance = 0 dengan Z-test for proportions (H0 = 1%)
# Berdasarkan analisis z-test, didapatkan p-value < 0,05 sehingga H0 ditolak dan hasilnya signifikan 
# Secara statistik jumlah trip = 0 sebesar 4.9% adalah signifikan . Sehingga hal ini menunjukkan adanya potensi anomali di data, pembatalan, dan lain-lain yang perlu digali lebih dalam 

from statsmodels.stats.proportion import proportions_ztest

count = df_cleaned['zero_trip_distance'].sum()

total_data = len(df_cleaned)

p0 = 0.01

z_stat, p_value = proportions_ztest(count, total_data, value=p0, alternative = 'larger')

print('p-value', p_value)


# In[43]:


# (6.2) Analisis rata-rata total_amount dengan rata- rata pickup hour 
# Didapatkan rata-rata total amount paling tinggi ada di menuju jam 05

df_cleaned['lpep_pickup_datetime'] = pd.to_datetime(df_cleaned['lpep_pickup_datetime'])

# Tambahkan kolom jam dan hari
df_cleaned['pickup_hour'] = df_cleaned['lpep_pickup_datetime'].dt.hour

# Rata-rata pickup hour
avg_pickup_hour = df_cleaned.groupby('pickup_hour')['total_amount'].mean()

plt.figure(figsize=(10,5))
avg_pickup_hour.plot(kind='line', marker='o', color='orange')
plt.title("Rata-rata Total Amount per Jam")
plt.xlabel("Jam (0-23)")
plt.ylabel("Rata-rata Total Amount (USD)")
plt.grid(True)
plt.show()
plt.savefig('Rata-rata Total Amount per Jam.png')   


# In[42]:


# (6.3) Analisis jumlah trip dengan rata- rata pickup hour 
# Didapatkan jumlah trip per jam paling banyak ada di range jam 15 - 20 

df_cleaned['pickup_hour'] = df_cleaned['lpep_pickup_datetime'].dt.hour

# Tambahkan jumlah_trip_per_hour
jumlah_trip_per_hour = df_cleaned.groupby('pickup_hour').size()
plt.figure(figsize=(8,5))
jumlah_trip_per_hour.plot(kind='line', marker='o', color='green')
plt.title("Jumlah Trip per Jam")
plt.xlabel("Jam (0-23)")
plt.ylabel("Jumlah Trips")
plt.grid(True)
plt.show()
plt.savefig('Rata-rata Jumlah trip per Jam.png')


# In[44]:


# (6.4) Top 10 pickup location
# Berdasarkan data pickup location didapatkan 10 lokasi dengan demand terbanyak sebagai berikut

top10_pickup = print(df_cleaned['PULocationID'].value_counts().head(10))


# In[49]:


# Plot bar chart
plt.figure(figsize=(8,5))
top10_pickup.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Top 10 Pickup Locations (by Trip Count)")
plt.xlabel("Pickup Location ID")
plt.ylabel("Jumlah Trips")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[40]:


# Kesimpulan dan Rekomendasi

# 1. Berdasarkan analisis z-test, didapatkan p-value < 0,05 sehingga secara statistik jumlah trip = 0 sebesar 4.9% adalah signifikan. 
# --> Adanya monitoring kepada driver setiap bulan untuk mengurangi kejadian trip distance = 0, dikarenakan ada potensi fraud, pembatalan dan lain -lain
# --> hal ini penting untuk mendorong productivity driver

# 2. Ditemukan rata-rata total amount paling tinggi ada di menuju jam 05 pagi. Hal ini menunjukkan adanya kontribusi pemasukan yang besar pada jam tersebut 
# --> Walaupun jumlah tripnya sedikit, tapi kontribusi per trip tinggi 
# --> Sehingga perlu dijaga loyalitas customer yang sudah terbiasa memakai jasa kita 
# --> Pendataan customer pagi untuk diberikan treatment khusus seperti diskon dan layanan express karena biasanya dibutuhkan cepat

# 3. Analisis jumlah trip dengan rata- rata pickup hour menunjukkan demand paling banyak berada pada range jam 15 - 20 
# --> Demand yang tinggi pada jam tersebut dapat menghasilkan revenue yang besar juga
# --> Memastikan jumlah supply driver yang lebih banyak pada jam tersebut dengan membuat shift driver  

# 4. Top 10 pickup location ada di lokasi 74, 75, 41, 166, 95, 82, 43, 97, 7, dan 244
# --> Penempatan driver dapat diperbanyak pada area yang memiliki banyak demand seperti diatas


# In[ ]:




