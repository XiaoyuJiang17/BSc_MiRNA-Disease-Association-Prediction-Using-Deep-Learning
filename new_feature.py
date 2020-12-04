import pandas as pd
import numpy as np

# Using miRNA-disease association matrix, miRNA functional similarity matrix,
# disease similarity matrix to construct new feature vector
data_md2 = pd.read_csv('data_md2.csv',header = None)
data_m = pd.read_csv('data_m.csv',header = None)
data_d = pd.read_csv('data_d.csv',header = None)
Headers_383 = np.arange(383)+1
Headers_495 = np.arange(495)+1

data_md2.columns , data_md2.index = Headers_495, Headers_383
data_m.columns, data_m.index = Headers_495, Headers_495
data_d.columns, data_d.index = Headers_383, Headers_383

print(data_md2)
print(data_m)
print(data_d)
# lebeled feature
data_label_feature = pd.DataFrame(columns = np.arange(878)+1, index = np.arange(5430)+1)
# for storing positions of miRNA and disease in feature set
# [j,i] j for disease , i for miRNA
data_label_feature_position = pd.DataFrame(columns = np.arange(2)+1, index = np.arange(5430)+1)
count = 1
for i in range(1,496):
    for j in range(1,384):
        if data_md2[i][j] != 0:
            data_label_feature.loc[count] = (data_m[i].append(data_d[j])).to_numpy()
            data_label_feature_position.loc[count] = [j,i]
            count += 1

data_label_feature.to_csv('data_label_feature.csv',header = None,index = None)
data_label_feature_position.to_csv('data_label_feature_position.csv',header = None, index = None)
print(data_label_feature)

# unlabeled fearure
data_unlabel_feature = pd.DataFrame(columns = np.arange(878)+1, index = np.arange(184155)+1)
# for storing positions of miRNA and disease in feature set
# [j,i] j for disease , i for miRNA
data_unlabel_feature_position = pd.DataFrame(columns = np.arange(2)+1, index = np.arange(184155)+1)
count = 1
for i in range(1,496):
    for j in range(1,384):
        if data_md2[i][j] == 0:
            data_unlabel_feature.loc[count] = (data_m[i].append(data_d[j])).to_numpy()
            data_unlabel_feature_position.loc[count] = [j,i]
            count += 1

data_unlabel_feature.to_csv('data_unlabel_feature.csv',header = None,index = None)
data_unlabel_feature_position.to_csv('data_unlabel_feature_position.csv',header = None, index= None)
print(data_unlabel_feature)
