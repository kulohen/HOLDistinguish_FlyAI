# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: wangyi
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
import numpy as np

'''
2019-8-15 
自定义结果weight 1:6:3
save by loss

2019-8-14 
添加一层dense
放弃dropout
平衡输入3类batch
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=200, type=int, help="train epochs")
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
hidden_dim = 512        # Dense层大小
max_seq_len = 34        # 最大句长
num_filters = 256       # 卷积核数目
kernel_size = 5         # 卷积核尺寸
learning_rate = 1e-3    # 学习率
numclass = 3            # 类别数
cw_train = {
    0:1,
    1:1,
    2:1
}
eval_weights = {
    0:1,
    1:6.,
    2:3.,
}
eval_weights_count = 10 # 应该是eval_weights的3个求和

model_cnn = Sequential()
model_cnn.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_len))
model_cnn.add(Conv1D(256, 5, activation='relu'))
model_cnn.add(MaxPooling1D(1))

model_cnn.add(Flatten())
model_cnn.add(Dense(hidden_dim, activation='relu'))
# model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(hidden_dim, activation='relu'))
# model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(numclass, activation='softmax'))

model_cnn.summary()
model_cnn.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=learning_rate),
              metrics=['acc'])

lr_level=2
best_score_by_acc = 0.
best_score_by_loss = 999.
x_train, y_train ,x_val, y_val= dataset.get_all_processor_data()
# x_val, y_val =dataset.get_all_validation_data()
for step in range(args.EPOCHS):
    cur_step = str(step + 1) + "/" + str(args.EPOCHS)
    print('\n步骤'+cur_step)

    x_3, y_3,x_4,y_4 =[],[],[],[]
    for iters in range(numclass):
        xx_tmp_train , yy_tmp_train, xx_tmp_val ,yy_tmp_val = dataset_slice[iters].next_batch()

        # 合并3类train
        x_3.append(xx_tmp_train)
        y_3.append(yy_tmp_train)
        x_4.append(xx_tmp_val)
        y_4.append(yy_tmp_val)
    x_3 = np.concatenate(x_3, axis=0)
    y_3 = np.concatenate(y_3, axis=0)
    x_4 = np.concatenate(x_4, axis=0)
    y_4 = np.concatenate(y_4, axis=0)

    this_epoch_loss_and_acc = model_cnn.fit(x=x_3, y=y_3, validation_data=(x_4, y_4),
                                            batch_size=args.BATCH ,epochs=1,verbose=2,
                                            class_weight=cw_train)

    sum_loss = 0.
    sum_acc = 0.
    for iters in range(numclass):
        history_test = model_cnn.evaluate(
            x=x_val_slice[iters],
            y=y_val_slice[iters],
            batch_size=None,
            verbose=2
        )
        print('class-%d __ loss :%.4f , acc :%.4f' %(iters ,history_test[0],history_test[1]))
        sum_loss += history_test[0] * eval_weights[iters]
        sum_acc += history_test[1] * eval_weights[iters]

    print('步骤 %d / %d: 自定义 val_loss is %.4f, val_acc is %.4f\n' %(step+1,args.EPOCHS, sum_loss/eval_weights_count , sum_acc/eval_weights_count))

    # save best loss
    if this_epoch_loss_and_acc.history['acc'][0] > 0.9 and best_score_by_loss >  sum_loss / eval_weights_count :
        model.save_model(model=model_cnn, path=MODEL_PATH, overwrite=True)
        best_score_by_acc = sum_acc / eval_weights_count
        best_score_by_loss = sum_loss / eval_weights_count
        print('【保存】了最佳模型by eval_loss : %.4f' %best_score_by_loss)

    if this_epoch_loss_and_acc.history['loss'][0] <0.3 and lr_level==2:
        model_cnn.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(lr=0.0001),
                      metrics=['accuracy'])
        print('【学习率】调整为 : 0,0001')
        lr_level = 3
    if this_epoch_loss_and_acc.history['loss'][0] < 0.15 and lr_level == 3:
        model_cnn.compile(loss='categorical_crossentropy',
                          optimizer=SGD(lr=0.00001),
                          metrics=['accuracy'])
        print('【学习率】调整为 : 0,00001')
        lr_level = 4
print('best_score_by_acc :%.4f' %best_score_by_acc)
print('best_score_by_loss :%.4f' %best_score_by_loss)