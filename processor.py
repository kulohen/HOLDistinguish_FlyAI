# -*- coding: utf-8 -*

import numpy
from data_helper import *
from flyai.processor.base import Base


MAX_LEN = 34


class Processor(Base):
    def __init__(self):
        super(Processor, self).__init__()
        self.word_dict = load_dict()
        self.count_processor = 0

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
        该方法字段与app.yaml中的input:->columns:对应
        '''

        sent_ids = word2id(text, self.word_dict, MAX_LEN)

        return sent_ids

    def input_y(self, label):
        '''
        参 数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。
        该方法字段与app.yaml中的output:->columns:对应
        '''
        # 0 - hate speech
        # 1 - offensive language
        # 2 - neither
        one_hot_label = numpy.zeros([3])  ##生成全0矩阵
        one_hot_label[label] = 1  ##相应标签位置置
        return one_hot_label

    def output_y(self, data):
        '''
        输出的结果，会被dataset.to_categorys(data)调用
        '''
        self.count_processor +=1
        # print('self.count_processor', self.count_processor)
        return numpy.argmax(data)
