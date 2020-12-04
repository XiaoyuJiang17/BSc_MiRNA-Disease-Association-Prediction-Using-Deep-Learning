import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime as dt
from sklearn import metrics
from matplotlib import pyplot as plt

A = pd.read_csv('data_md2.csv', header=None).to_numpy()  # (383,495)
SM = pd.read_csv('data_m.csv', header=None).to_numpy()
SD = pd.read_csv('data_d.csv', header=None).to_numpy()
labeled_data_position = pd.read_csv('data_label_feature_position.csv', header=None).to_numpy()


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


## Build the encoder
latent_dim = 100

encoder_inputs = keras.Input(shape=(878,))
x = layers.Dense(300, activation="relu")(encoder_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
# encoder.summary()

## Build the decoder

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(300, activation="relu")(latent_inputs)
decoder_outputs = layers.Dense(878, activation="relu")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")


# decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 878
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def Training_set_generation(i):  # i is the index of sample
    d = labeled_data_position[i][0] - 1  # 0-382
    m = labeled_data_position[i][1] - 1  # 0-494
    SSM = np.concatenate((A.T, SM), axis=1)
    SSM_l = np.delete(SSM,m,axis=0)
    SSD = np.concatenate((A, SD), axis=1)
    SSD_l = np.delete(SSD,d,axis=0)
    return d, m, SSM, SSD,SSM_l,SSD_l


def predicting(d,SSM,SSD,SSM_l,SSD_l): # SSM_l,SSD_l are used for training, while SSM,SSD used for testing
    vae1 = VAE(encoder, decoder)
    vae1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    vae1.fit(SSM_l, epochs=50, batch_size=20)
    vae2 = VAE(encoder, decoder)
    vae2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    vae2.fit(SSD_l, epochs=50, batch_size=20) #fitting uses 5429 samples
    predict1 = vae1.decoder(vae1.encoder(SSM))
    predict_1 = predict1[:, d].numpy()
    predict2 = vae2.decoder(vae2.encoder(SSD))
    predict_2 = predict2[d, :495].numpy()
    predict_value = 0.5 * (predict_1 + predict_2)
    return predict_value


def new_predict_generation(d, predict):  # disease 0-382
    association_list = np.array([])
    new_predict = np.array([])
    for j in range(0, 5430):
        if labeled_data_position[j][0] - 1 == d:  # association with same disease
            association_list = np.append(association_list, labeled_data_position[j][1])  # value from 1- 495)
    for num in range(0, 495):
        if num not in association_list - 1:
            new_predict = np.append(new_predict, predict[num])
    return new_predict


def Rank_function(scores):  # values start from 1
    Scores_df = pd.DataFrame(scores)
    Sorted_Scores = Scores_df.sort_values(ascending=True, by=0)  # 从小到大排序，将返回对series赋值给a
    b = Sorted_Scores.index.to_numpy()  # 取a对index并转换为numpy的格式,from 0 to max
    count = 1
    Rank_1 = np.zeros(len(scores))  # ranking start from 1
    for j in b:
        Rank_1[j] = count
        count = count + 1
    return Rank_1


Positive_index = []
Candidate_index = []
start_time = dt.datetime.now()
for i in range(0, 5):  # 5430
    d, m, SSM,SSD,SSM_l,SSD_l = Training_set_generation(i)
    predict = predicting(d, SSM, SSD,SSM_l,SSD_l)  # (495,)
    new_predict = new_predict_generation(d, predict)  # delete know association
    new_predict2 = np.append([predict[m]], new_predict)  # add test association
    Rank = Rank_function(new_predict2)
    ind_test_sample, ind_candi_sample = 0, np.random.randint(1, len(new_predict2))
    Positive_index.append(Rank[ind_test_sample] / len(new_predict2))
    Candidate_index.append(Rank[ind_candi_sample] / len(new_predict2))
    print(i, ' Pos ',Rank[ind_test_sample] / len(new_predict2),' Can ',Rank[ind_candi_sample] / len(new_predict2))
end_time = dt.datetime.now()
elapsed_time = end_time - start_time
print('elapsed_time', elapsed_time)
print('Pos', Positive_index)
print('Can', Candidate_index)
fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((np.ones(5), np.zeros(5)), axis=0),
                                         np.concatenate((Positive_index, Candidate_index), axis=0))
roc_auc = metrics.auc(fpr, tpr)  ###计算auc的值

lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, marker='o', color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()