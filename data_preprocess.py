import numpy as np
import pandas as pd
import math

## runing this piece of code, three csv files will be generated: data_md2, data_m and data_d

#construct the association matrix data_md2
data_md = pd.read_csv('miRNA-disease association data.csv',header = None)
headers = ["miRNA","disease"]
data_md.columns = headers
data_md2 = pd.DataFrame(columns=np.arange(495)+1, index=np.arange(383)+1)
for tup in data_md.itertuples():
    data_md2[tup[1]][tup[2]] = 1
    #num of miRNA is tup[1]
    #num of disease is tup[2]
data_md2.replace(np.nan,0,inplace = True)
data_md2.to_csv('data_md2.csv',header = None, index = None)

#construct Gaussian interaction profile kernel similarity for miRNAs(data_km)
data_km = pd.DataFrame(columns = np.arange(495)+1, index = np.arange(495)+1)
gamma_mm = 1
itr = np.arange(495)+1
s = 0
for i in itr:
    a = data_md2[i].to_numpy()
    s = s + (a**2).sum()
gamma_m = gamma_mm * 495 / s

for col in range(1,496):
    for ind in range(1,496):
        a = (data_md2[col] - data_md2[ind])
        data_km[col][ind] = math.exp( -gamma_m * (a**2).sum() )
#print(data_km.head())

#construct Gaussian interaction profile kernel similarity for disease(data_kd)
data_kd = pd.DataFrame(columns = np.arange(383)+1, index = np.arange(383)+1)
gamma_dd = 1
itr = np.arange(383)+1
s = 0
for i in itr:
    a = data_md2[:][i].to_numpy()
    s = s + (a**2).sum()
gamma_m = gamma_mm * 383 / s
#print(s)
for col in range(1,384):
    for ind in range(1,384):
        a = (data_md2.loc[col] - data_md2.loc[ind])
        data_kd[col][ind] = math.exp( -gamma_m * (a**2).sum() )

#print(data_kd.head())

#Integrated similarity for miRNAs (data_m)
data_m = pd.DataFrame(columns = np.arange(495)+1, index = np.arange(495)+1)
data_sm = pd.read_csv('functional similarity matrix.csv',header = None)

headers = np.arange(1,496)
data_sm.columns = headers
data_sm.index = headers

for i in range(1,496):
    for j in range (1,496):
        if data_sm[i][j] > 0:
            data_m[i][j] = data_sm[i][j]
        else:
            data_m[i][j] = data_km[i][j]

#print('data_m ', data_m)
data_m.to_csv('data_m.csv',header = None, index = None)

#Integrated similarity for diseases(data_d)
data_d = pd.DataFrame(columns = np.arange(383)+1, index = np.arange(383)+1)
data_sd1 = pd.read_csv('disease semantic similarity matrix 1.csv',header = None)
data_sd2 = pd.read_csv('disease semantic similarity matrix 2.csv',header = None)

headers = np.arange(1,384)
data_sd1.columns = headers
data_sd1.index = headers
data_sd2.columns = headers
data_sd2.index = headers

data_sd = 0.5 * ( data_sd1 + data_sd2 )
for i in range(1,384):
    for j in range (1,384):
        if data_sd[i][j] > 0:
            data_d[i][j] = data_sd[i][j]
        else:
            data_d[i][j] = data_kd[i][j]

data_d.to_csv('data_d.csv',header = None, index = None)