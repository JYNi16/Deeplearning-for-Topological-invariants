import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, losses
import glob, os

class HamData():
    def __init__(self, path):
        self.list = []
        self.list.extend(glob.glob(os.path.join(path, "*.npz")))
        self.batch_size = 50

    def load_data(self, path_l):
        Ham = np.load(path_l.numpy())
        data = np.reshape(Ham["s"].astype(np.float32), [33, 2, -1])
        wn = Ham["label"].astype(np.float32)
        if wn == "0":
            label = 0
        elif wn == "1":
            label = 1
        else:
            label = 2

        return data, label


    def createdata(self):
        data_train = self.list
        db_train = tf.data.Dataset.from_tensor_slices((data_train))
        train_data = db_train.map(lambda path_l :tf.py_function(self.load_data,inp=[path_l],
                                                                Tout=[tf.float32, tf.int32]),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data = train_data.batch(self.batch_size)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        return train_data

if __name__=="__main__":
    path = "E:/deeplearning\depp learning for Phys/Deep-Learning-Topological-Invariants/train_data1"
    data = HamData(path)
    db_train = data.createdata()
    for x, y in db_train:
        print(x.shape)

