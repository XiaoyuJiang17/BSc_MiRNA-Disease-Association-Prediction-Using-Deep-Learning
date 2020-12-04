import tensorflow as tf
import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime as dt

# version two: The test sample's feature is not given to the model.
#Warning : original paper use linear kernal in svm
start_time = dt.datetime.now()
encoder = tf.keras.models.load_model('/tmp/model')

label_feature = pd.read_csv('data_label_feature.csv',header = None)
unlabel_all = pd.read_csv('data_unlabel_feature.csv',header = None)
label_feature_position = pd.read_csv('data_label_feature_position.csv',header = None)
unlabel_feature_position = pd.read_csv('data_unlabel_feature_position.csv',header = None)
labeled_sample = label_feature.to_numpy() #Positive associations
unlabel_sample_all = unlabel_all.to_numpy()

def trainset_generation(num): #num: 0 - 5430
    unlabel_sample = unlabel_all.sample(n=5430, random_state = num, replace=False, axis=0).to_numpy()
    labeled_sample_temp = np.delete(labeled_sample,num,axis = 0)
    Train = np.concatenate((labeled_sample_temp, unlabel_sample), axis=0)  # (10859,878)Training set features
    return Train

def label_generation(i): #i:0-5430
    Positive_labels = np.ones(5429)
    Negative_labels = np.zeros(5430)
    labels_svm = np.concatenate((Positive_labels, Negative_labels), axis=0)  # training set labels
    return labels_svm

def related_association_finding(i):
    j = label_feature_position[0][i]  # disease j (in range 1-383)
    Testing_set_1 = labeled_sample[i]  # i-th sample, before reshape
    Testing_set = Testing_set_1.reshape(-1, 878)
    for num in range(0, 184155):
        if unlabel_feature_position[0][num] == j:
            # list_num = list_num.append(num)
            x = unlabel_sample_all[num].reshape(-1, 878)
            Testing_set = np.concatenate((Testing_set, x), axis=0)  # predicting(Testing) set
    return Testing_set

def train_predict(Train,Train_label,Predict):
    Train_encoded = encoder.predict(Train)
    Predict_encoded = encoder.predict(Predict)
    # Building svm model and fitting
    model_LinearSVC = svm.LinearSVC()
    model_LinearSVC.fit(Train_encoded,Train_label)
    # Predicting Scores
    Scores = model_LinearSVC.decision_function(Predict_encoded)
    return Scores

def Rank_function(scores): #values start from 1
    Scores_df = pd.DataFrame(scores)
    Sorted_Scores = Scores_df.sort_values(ascending = True, by = 0)  # 从小到大排序，将返回对series赋值给a
    b = Sorted_Scores.index.to_numpy()  # 取a对index并转换为numpy的格式,from 0 to max
    count = 1
    Rank_1 = np.zeros(len(scores)) #ranking start from 1
    for j in b:
        Rank_1[j] = count
        count = count + 1
    return Rank_1

# evaluation, building training and testing sets
Positive_index = []
Candidate_index = []

for i in range(0,24): # 0 - 5430
    Train = trainset_generation(i)
    labels_svm = label_generation(i) # training set labels
    Testing_set = related_association_finding(i) # figure out related UNKNOWN association!
    scores = train_predict(Train,labels_svm,Testing_set)
    Rank = Rank_function(scores)
    ind_test_sample, ind_candi_sample = 0, np.random.randint(1,len(scores))
    Positive_index.append(Rank[ind_test_sample]/len(scores))
    Candidate_index.append(Rank[ind_candi_sample]/len(scores))
end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print(elapsed_time)

labels = np.concatenate((np.ones(24),np.zeros(24)),axis = 0) #5430
predict_score_svm = np.concatenate((np.array(Positive_index),np.array(Candidate_index)),axis = 0)
fpr, tpr, thresholds = metrics.roc_curve(labels, predict_score_svm)
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




