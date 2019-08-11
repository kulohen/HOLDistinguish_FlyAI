# -*- coding: utf-8 -*
import os
import tensorflow as tf
from flyai.model.base import Base
from path import MODEL_PATH
from tensorflow.python.saved_model import tag_constants
from keras.engine.saving import load_model

KERAS_MODEL_NAME = "model.h5"

class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        self.count_predict = 0

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        if self.model is None:
            self.model = load_model(self.model_path)
        data = self.model.predict(self.dataset.predict_data(**data))
        data = self.dataset.to_categorys(data)
        self.count_predict +=1
        # print('self.count_predict' ,self.count_predict)

        return data

    def predict_all(self, datas):
        if self.model is None:
            self.model = load_model(self.model_path)
        labels = []
        for data in datas:
            data = self.model.predict(self.dataset.predict_data(**data))
            data = self.dataset.to_categorys(data)
            labels.append(data)
        print('predict datas : ', len(labels))
        return labels

    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        model.save(os.path.join(path, name))

