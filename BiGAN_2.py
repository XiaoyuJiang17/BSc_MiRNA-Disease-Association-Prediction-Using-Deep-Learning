from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

X_train = pd.read_csv('data_label_feature.csv',header = None).to_numpy() # feature with known association
Unlabeled = pd.read_csv('data_unlabel_feature.csv',header = None).to_numpy() # feature with unknown association
labeled_position = pd.read_csv('data_label_feature_position.csv',header = None) # association position
unlabel_position = pd.read_csv('data_unlabel_feature_position.csv',header = None)

class BIGAN():
    def __init__(self,Train):
        self.Train = Train
        self.dimension = 878
        self.img_shape = (self.dimension, ) # input shape, that is the combination of miRNA and disease
        self.latent_dim = 100

        optimizer = Adam(learning_rate = 0.0001) 

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)


    def build_encoder(self):
        '''model = Sequential()

        model.add(Dense(400, input_shape = self.img_shape))
        model.add(LeakyReLU(alpha = 0.2))
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z) '''

        inputs = tf.keras.Input(shape=(self.dimension,))
        x = tf.keras.layers.Dense(400, activation=tf.nn.leaky_relu)(inputs)
        x = tf.keras.layers.Dense(200,activation=tf.nn.leaky_relu)(x)
        outputs = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.leaky_relu)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def build_generator(self):
        '''model = Sequential()

        model.add(Dense(400, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(878))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)'''
        inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu)(inputs)
        x = tf.keras.layers.Dense(400,activation=tf.nn.leaky_relu)(x)
        outputs = tf.keras.layers.Dense(self.dimension, activation=tf.nn.tanh)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=(self.dimension,))
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(400)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        #model = Dropout(0.5)(model)
        model = Dense(400)(model)
        model = LeakyReLU(alpha=0.2)(model)
        #model = Dropout(0.5)(model)
        model = Dense(400)(model)
        model = LeakyReLU(alpha=0.2)(model)
        #model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)
        # 0 for fake and 1 for real
        return Model([z, img], validity)

        '''inputs1 = tf.keras.Input(shape=(self.latent_dim,))
        inputs2 = tf.keras.Input(shape = (self.dimension,))
        inputs = concatenate([inputs1,inputs2])
        x = tf.keras.layers.Dense(400, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(1, activation=tf.nn.relu)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model'''


    def train(self, epochs, batch_size=128):

        '''# Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)'''

        #X_train = pd.read_csv('data_label_feature.csv',header = None).to_numpy()


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        #D_loss_list = []
        #G_loss_list = []
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(batch_size, self.latent_dim))
            imgs_ = self.generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, self.Train.shape[0], batch_size)
            imgs = self.Train[idx]
            z_ = self.encoder.predict(imgs)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])

            # Plot the progress
            # D_loss_list.append(d_loss[0])
            # G_loss_list.append(g_loss[0])
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

        #plt.plot(D_loss_list)
        #plt.title('D loss')
        #plt.show()
        #plt.plot(G_loss_list)
        #plt.title('G loss')
        #plt.show()




def related_association_finding(i): # i is the position of labeled feature
    j = labeled_position[0][i]  # disease j (in range 1-383)
    Testing_set_1 = X_train[i]  # i-th sample, before reshape
    Testing_set = Testing_set_1.reshape(-1, 878)
    for num in range(0, 184155):
        if unlabel_position[0][num] == j:
            # list_num = list_num.append(num)
            x = Unlabeled[num].reshape(-1, 878)
            Testing_set = np.concatenate((Testing_set, x), axis=0)  # predicting(Testing) set
    return Testing_set

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


def Score_function(Testing_one):
    theta = 50
    encodered = tf.constant(bigan.encoder.predict(Testing_one))
    gen = bigan.generator(encodered)
    b = tf.reshape(gen, [Testing_one.shape[0], 878])
    distance = tf.square(b - Testing_one)
    ones = tf.ones([878, 1], dtype=tf.float32)
    dis_add1 = tf.matmul(distance, ones)
    dis_add2 = abs(1 - bigan.discriminator([encodered,Testing_one]).numpy())
    dis_add = dis_add1 + theta * dis_add2
    return dis_add

Positive_index = []
Candidate_index = []
num_folds = 5

if __name__ == '__main__':
    for num in range(0,num_folds):# 0 - num_folds
        start = int( num * 5430 / num_folds)
        end = int( ( num + 1 ) * 5430 / num_folds )
        #Testing_set = X_train[start:end] #(1086, 878)
        Training_set = np.delete(X_train, np.arange(1086) + start, axis = 0) # (4344, 878)
        bigan = BIGAN(Training_set)
        bigan.train(epochs = 15000, batch_size=32)
        for i in range(start,end): #end
            Testing_one = related_association_finding(i) # Testing_one consists of one test sample and all related unknown association
            Score = -1 * Score_function(Testing_one) # larger prediction means larger error, should be lower rank
            Rank = Rank_function(Score)
            ind_test_sample, ind_candi_sample = 0, np.random.randint(1, len(Score))
            Positive_index.append(Rank[ind_test_sample] / len(Score))
            Candidate_index.append(Rank[ind_candi_sample] / len(Score))
            print("%d [PosRank: %f, PosNum: %f] [CanRank: %f, CanNum: %f]" %
                  (i, Rank[ind_test_sample]/len(Score),Score[ind_test_sample],Rank[ind_candi_sample]/len(Score),Score[ind_candi_sample] ))
    label = np.concatenate((np.ones(5430),np.zeros(5430)),axis = 0)
    predict = np.concatenate((Positive_index,Candidate_index),axis = 0)
    fpr, tpr, thresholds = metrics.roc_curve(label,predict)
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


