# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import tensorflow as tf
from flyai.dataset import Dataset
from model import Model as flyai_model
from path import MODEL_PATH, LOG_PATH
import WangyiUtilOnFlyai as wangyi
from keras.layers import Input, Dense , Embedding,Conv1D ,MaxPooling1D ,Flatten ,Dropout
from keras.models import Model ,Sequential
from keras.optimizers import SGD,Adam,RMSprop

'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=20, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = flyai_model(dataset)
print('dataset.get_train_length()',dataset.get_train_length())
print('dataset.get_validation_length()',dataset.get_validation_length())
dataset_slice = wangyi.getDatasetListByClassfy(classify_count=3)
x_train_slice, y_train_slice, x_val_slice, y_val_slice = [], [], [], []
for epoch in range(3):
    x_1, y_1, x_2, y_2 = dataset_slice[epoch].get_all_processor_data()
    x_train_slice.append(x_1)
    y_train_slice.append(y_1)
    x_val_slice.append(x_2)
    y_val_slice.append(y_2)

# 超参
vocab_size = 20655      # 总词汇量
embedding_dim = 64      # 嵌入层大小
hidden_dim = 1024        # Dense层大小
max_seq_len = 34        # 最大句长
num_filters = 256       # 卷积核数目
kernel_size = 5         # 卷积核尺寸
learning_rate = 1e-3    # 学习率
numclass = 3            # 类别数


model_cnn = Sequential()
model_cnn.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_len))
model_cnn.add(Conv1D(256, 5, activation='relu'))
model_cnn.add(MaxPooling1D(1))

model_cnn.add(Flatten())
model_cnn.add(Dense(hidden_dim, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(numclass, activation='softmax'))

model_cnn.summary()
model_cnn.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=learning_rate),
              metrics=['acc'])

best_score_by_acc = 0.
best_score_by_loss = 999.
x_val, y_val =dataset.get_all_validation_data()
for step in range(dataset.get_step()):
    cur_step = str(step + 1) + "/" + str(dataset.get_step())
    print('\n步骤'+cur_step)
    x_train, y_train = dataset.next_train_batch()

    this_epoch_loss_and_acc = model_cnn.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=1,verbose=2)
    for iters in range(numclass):
        history_test = model_cnn.evaluate(
            x=x_val_slice[iters],
            y=y_val_slice[iters],
            batch_size=None,
            verbose=2
        )
        print('class-%d __ loss :%.4f , acc :%.4f' %(iters ,history_test[0],history_test[1]))
    # save best acc
    if this_epoch_loss_and_acc.history['loss'][0] <0.1 and best_score_by_acc <  this_epoch_loss_and_acc.history['val_acc'][0] :
        model.save_model(model=model_cnn, path=MODEL_PATH, overwrite=True)
        best_score_by_acc = this_epoch_loss_and_acc.history['val_acc'][0]
        best_score_by_loss = this_epoch_loss_and_acc.history['val_loss'][0]
        print('【保存】了最佳模型by val_acc : %.4f' %best_score_by_acc)

