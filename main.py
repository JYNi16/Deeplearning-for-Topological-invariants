import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import CNN_TI
from keras import layers, optimizers, metrics, losses
from dataloader import HamData


# net = tf.keras.Sequential()
# net.add(layers.Conv2D(40, kernel_size=(2, 2), activation='relu', input_shape=(11, 6, 1)))
# #model.add(MaxPooling2D(pool_size=(2,2)))
# net.add(layers.Conv2D(1, kernel_size=(1, 1), activation='relu'))
# net.add(layers.Flatten())
# net.add(layers.Dense(2, activation='relu'))
# net.add(layers.Dense(1, activation='linear'))
#
model = CNN_TI()
#
# def train(train_data):
#     loss_all = 0
#     for epoch in range(25):
#         for step, data in enumerate(train_data):
#             with tf.GradientTape() as tape:
#                 x, y = data
#                 x = tf.reshape(x, [-1, 12, 12, 1])
#                 out = model(x)
#                 y_onehot = tf.one_hot(y, depth=3)
#                 loss = tf.keras.losses.categorical_crossentropy(from_logits=True, y_true=y_onehot, y_pred=out)
#                 loss = tf.reduce_mean(loss)
#                 grads = tape.gradient(loss, model.trainable_variables)
#                 optimizer.apply_gradients(zip(grads, model.trainable_variables))
#                 acc_meter.update_state(tf.argmax(out, axis=1), y)
#             if (step % 10 == 0):
#                 print("step is:", step)
#         print("epoch:", epoch, "| Loss is:%.4f" % (loss), "|Accuracy is:%.4f" % (acc_meter.result().numpy()))
#         acc_meter.reset_states()
#     tf.saved_model(model, "saved/1")


def train_wn(train_data):
    optimizer = optimizers.Adam(lr=0.001)
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    for epoch in range(25):
        loss_all = 0
        acc = 0
        for step, data in enumerate(train_data):
            with tf.GradientTape() as tape:
                x, y = data
                x = tf.reshape(x, [-1, 11,6, 1])
                out = model(x)
                loss = losses.mean_squared_error(y, out)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_all += loss
            acc += 1 - loss
            #print("acc is:", acc)
            if (step % 100 == 0):
                print("step is:", step)
        print("epoch:", epoch, "| Loss is:%.4f" % (loss_all/step), "| Acc is:%.4f"%(acc/step))


if __name__=="__main__":
    path = "E:/deeplearning\depp learning for Phys/Deep-Learning-Topological-Invariants/train_data1"
    data = HamData(path)
    db_train = data.createdata()
    train_wn(db_train)