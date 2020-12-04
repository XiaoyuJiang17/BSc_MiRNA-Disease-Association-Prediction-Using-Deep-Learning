import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import datetime as dt
import copy

#V2: In testing, each tested mirna-disease associaton will be ranked with candidate mirnas with same disease, for each prediction
# only other association information is considered!
#V2  AUC : 0.78

A = pd.read_csv('data_md2.csv',header = None).to_numpy()  # index from 0
temp_A = pd.DataFrame(A) # same matrix A stored in pandas format, which is useful format for getting rows and columns
temp_A.columns, temp_A.index = np.arange(495) + 1, np.arange(383) + 1

data_m  = pd.read_csv('data_m.csv',header = None)
data_d = pd.read_csv('data_d.csv',header = None)
Header_383 = np.arange(383)+1
Header_495 = np.arange(495)+1
data_m.columns,data_m.index = Header_495, Header_495
data_d.columns,data_d.index = Header_383, Header_383

labeled_position = pd.read_csv('data_label_feature_position.csv',header = None).to_numpy()

# Within_m
def Within_m_one(m, d):
    # for one particular mirna-disease pair, compute its within_m value
    # m: tested miRNA(1-495),d: tested disease()
    lists = []
    a = temp_A.loc[d]  # 0 or 1 vector, aiming to find 1s
    for i in range(1, 496):
        if a[i] == 1 and i != m:
            lists.append(i) # store miRNAs which has association with d(in the dth row,except the tested miRNA)
    lists2 = [0]  # prevent zero size array
    for num in lists:
        lists2.append(data_m[m][num])
    x = np.array(lists2).max()
    return x

def Within_m_seires(d): # for given disease, compute values of within_m of all mirnas
    Within_m = pd.Series(index=np.arange(495) + 1, dtype=float)
    for m in range(1,496):
        Within_m[m] = Within_m_one(m,d)
    return Within_m

# Within_d
def Within_d_one(m,d):
    # for one particular mirna-disease pair,compute its within_d value
    lists = []
    a = temp_A.loc[:, m]
    for i in range(1, 384):
        if a[i] == 1 and i != d:
            lists.append(i) #store diseases which has association with m
    lists2 = [0]
    for num in lists:
        lists2.append(data_d[d][num])
    x = np.array(lists2).max()
    return x

def Within_d_series(d): # for given disease, compute values of within_d of all mirnas
    Within_d = pd.Series(index=np.arange(495) + 1, dtype=float)
    for m in range(1,496):
        Within_d[m] = Within_d_one(m,d)
    return Within_d

# Between_m
def Between_m_one(m,d):
    lists = []
    a = temp_A.loc[d]  # getting the dth row of temp_A matrix, 0 or 1 vector, aiming to find 0s
    for i in range(1, 496):
        if a[i] == 0 and i != m:
            lists.append(i)
    lists2 = [0]
    for num in lists:
        lists2.append(data_m[m][num])
    x = np.array(lists2).max()
    return x

def Between_m_series(d):
    Bewteen_m = pd.Series(index=np.arange(495) + 1, dtype=float)
    for m in range(1,496):
        Bewteen_m[m] = Between_m_one(m,d)
    return Bewteen_m

# Between_d
def Between_d_one(m,d):
    lists = []
    a = temp_A.loc[:, m]  # 0 or 1 vector, aiming to find 0s
    for i in range(1, 384):
        if a[i] == 0 and i != d:
            lists.append(i)
    lists2 = [0]
    for num in lists:
        lists2.append(data_d[d][num])
    x = np.array(lists2).max()
    return x

def Between_d_series(d):
    Between_d = pd.Series(index=np.arange(495) + 1, dtype=float)
    for m in range(1,496):
        Between_d[m] = Between_m_one(m,d)
    return Between_d

# Final Scores
# giving a disease d(1-383), get the predicted F_scores for all related miRNAs
def get_F(d): #(495,) return: series,index 1 -495
    F = Within_m_seires(d) * Within_d_series(d) / Between_d_series(d) * Between_m_series(d)
    return F

def Rank_function(F): #values start from 1
    Scores_df = pd.DataFrame(F)
    Sorted_Scores = Scores_df.sort_values(ascending = True, by = 0)  # 从小到大排序，将返回对series赋值给a
    b = Sorted_Scores.index.to_numpy()  # 取a对index并转换为numpy的格式,from 0 to max
    count = 1
    Rank_1 = np.zeros(len(F)) #ranking start from 1
    for j in b:
        Rank_1[j] = count
        count = count + 1
    return Rank_1

def d_m_finding(i):
    d = labeled_position[i][0]   # 1 - 383
    m = labeled_position[i][1]   # 1 - 495
    return d,m

def new_F_generation(d,F): # disease 1-383, remove untested KNOWN associations
    association_list = np.array([])
    new_F = np.array([])
    for j in range(0, 5430):
        if labeled_position[j][0]  == d:  # association with same disease
            association_list = np.append(association_list, labeled_position[j][1])  # value from (1- 495)
    for num in range(0, 495): # here because in main for loop, F has been converted to numpy format
        if num not in association_list - 1:
            new_F = np.append(new_F, F[num])
    return new_F

# for each tested association
# get disease and miRNA index for 1th association
# get predicted F_scores for this disease(495,)
# new_predicted generation: test known association + candidate association(remove untested KNOWN associations)
# ranking for testes association and random selected candidate
# store their ranking
Positive_index = []
Candidate_index = []
start_time = dt.datetime.now()
for i in range(3210,3216):  # 0 - 5430
    d,m = d_m_finding(i) # d:1-383,m:1-495
    F_score = get_F(d) # series,index 1 - 495
    F_score = F_score.to_numpy() # numpy,0-494
    new_F1 = new_F_generation(d,F_score) # delete all known association
    new_F = np.concatenate(([F_score[m-1]],new_F1),axis = 0)
    Rank = Rank_function(new_F)
    ind_test_sample = 0
    ind_candi_sample = np.random.randint(1,len(new_F))
    Positive_index.append(Rank[ind_test_sample] / len(new_F))
    Candidate_index.append(Rank[ind_candi_sample] / len(new_F))
    print(i," Pos ",Rank[ind_test_sample] / len(new_F)," Can ",Rank[ind_candi_sample] / len(new_F))
end_time = dt.datetime.now()
time_elasped = end_time - start_time
print('time_elasped',time_elasped)
print(Positive_index)
print(Candidate_index)
fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((np.ones(6),np.zeros(6)),axis = 0),
                                         np.concatenate((Positive_index,Candidate_index),axis = 0))
roc_auc = metrics.auc(fpr, tpr)  ###计算auc的值

lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, marker = 'o',color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





