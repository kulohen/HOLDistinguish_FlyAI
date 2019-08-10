#!/usr/bin/env python
# coding:utf-8
"""
Name : WangyiUtilOnFlyai.py
Author  : 莫须有的嚣张
Contect : 291255700
Time    : 2019/7/28 上午9:42
Desc:
"""

import sys
from time import time

import json
import os
import platform
import random
import requests
from flyai.dataset import Dataset
from flyai.source.source import Source
from flyai.source.base import Base, DATA_PATH
from flyai.utils.yaml_helper import Yaml
import pandas as pd

# TODO 继承Source类
class SourceByWangyi(Source):
    def __init__(self, custom_source=None):
        yaml = Yaml()
        try:
            f = open(os.path.join(sys.path[0], 'train.json'))
            line = f.readline().strip()
        except IOError:
            line = ""

        postdata = {'id': yaml.get_data_id(),
                    'env': line,
                    'time': time(),
                    'sign': random.random(),
                    'goos': platform.platform()}
        try:
            try:
                servers = yaml.get_servers()
                r = requests.post(servers[0]['url'] + "/dataset", data=postdata)
                self.__source = json.loads(r.text)
            except:
                self.__source = None

            if self.__source is None:
                self.__source = self.create_instance("flyai.source.csv_source", 'Csv',
                                                     {'train_url': os.path.join(DATA_PATH, "dev.csv"),
                                                      'test_url': os.path.join(DATA_PATH, "dev.csv")}, line)
            elif 'yaml' in self.__source:
                self.__source = self.__source['yaml']
                if custom_source is None:
                    self.__source = self.create_instance("flyai.source." + self.__source['type'] + "_source",
                                                         self.__source['type'].capitalize(), self.__source['config'],
                                                         line)
                else:
                    self.__source = custom_source
            else:
                if not os.path.exists(os.path.join(DATA_PATH, "train.csv")) and not os.path.exists(
                        os.path.join(DATA_PATH, "test.csv")):
                    raise Exception("invalid data id!")
                else:
                    self.__source = self.create_instance("flyai.source.csv_source", 'Csv',
                                                         {'train_url': os.path.join(DATA_PATH, "train.csv"),
                                                          'test_url': os.path.join(DATA_PATH, "test.csv")}, line)
        except TypeError:
            pass
        self.source_csv = self.__source

    def get_sliceCSVbyClassify(self,label='label',classify_count=3):
        # step 1 : csv to dataframe
        dataframe_train = pd.DataFrame(data=self.source_csv.data)
        dataframe_test = pd.DataFrame(data=self.source_csv.val)

        # step 2 : 筛选 csv
        list_path_train,list_path_test = [],[]
        for epoch in range(classify_count):
            path_train = os.path.join(DATA_PATH, 'wangyi-train-classfy-' + str(epoch) + '.csv')
            dataframe_train[dataframe_train[label] == epoch].to_csv(path_train,index=False)
            list_path_train.append(path_train)

            path_test = os.path.join(DATA_PATH, 'wangyi-test-classfy-' + str(epoch) + '.csv')
            dataframe_test[dataframe_test[label] == epoch].to_csv(path_test,index=False)
            list_path_test.append(path_test)

        return list_path_train,list_path_test


def readCsv_onFlyai(readCsvOnLocal=True):
    try:
        f = open(os.path.join(sys.path[0], 'train.json'))
        line = f.readline().strip()
    except IOError:
        line = ""

    if readCsvOnLocal:
        source_csv = Source().create_instance("flyai.source.csv_source", 'Csv',
                                              {'train_url': os.path.join(DATA_PATH, "dev.csv"),
                                               'test_url': os.path.join(DATA_PATH, "dev.csv")}, line)
    else:
        source_csv = Source().create_instance("flyai.source.csv_source", 'Csv',
                                              {'train_url': os.path.join(DATA_PATH, "train.csv"),
                                               'test_url': os.path.join(DATA_PATH, "test.csv")}, line)
    # 实际上返回的是 flyai.source.csv_source.py的 Csv类, source_csv.data 是train_csv文件,source_csv.val 是test_csv文件
    dataframe_train = pd.DataFrame(data=source_csv.data)
    dataframe_test = pd.DataFrame(data=source_csv.val)

    print('train data 透视表')
    print(pd.pivot_table(dataframe_train, values=['label'], index=['label'], aggfunc='count'))
    print('test data 透视表')
    print(pd.pivot_table(dataframe_test, values=['label'], index=['label'], aggfunc='count'))

    return source_csv


def readCustomCsv(train_csv_url, test_csv_url):
    try:
        f = open(os.path.join(sys.path[0], 'train.json'))
        line = f.readline().strip()
    except IOError:
        line = ""

    source_csv = Source().create_instance("flyai.source.csv_source", 'Csv',
                                          {'train_url': os.path.join(DATA_PATH, train_csv_url),
                                           'test_url': os.path.join(DATA_PATH, test_csv_url)}, line)

    return source_csv

def ExtendDataFrameToSize(dataframe,size):
    '''

    :param dataframe: 要被扩展的csv
    :param size: 扩展到指定size的大小。
    :return: 比如原csv是12行，size输入25，那么csv至少扩展到25以上，即是12 *2 *2 =48，返回48的大小。
    '''
    if dataframe.shape[0] == size :
        return dataframe
    elif dataframe.shape[0] > size :
        # 裁剪csv到size大小
        return dataframe[0:size]
    else:
        dataframe = pd.concat([dataframe,dataframe.copy(deep=False)])
        return ExtendDataFrameToSize(dataframe, size)

def ExtendCsvToSize(source_csv , label='label', size=-1 ,classify_count = -1):
    '''

    :param source_csv: dataframe格式，数据源
    :param label: 筛选的label
    :param size: 定义需要扩容的size
    :return: 扩容后的CSV，dataframe格式
    '''
    df5 = pd.DataFrame()
    for i in range(classify_count):
        if source_csv[source_csv[label] == i].empty :
            break
        tmp = ExtendDataFrameToSize( source_csv[source_csv[label] == i], size)
        df5 = pd.concat([df5, tmp])
    return df5


def DatasetExtendToSize(readCsvOnLocal=True , train_size=32 ,val_size=32,classify_count=10):
    '''

    :param readCsvOnLocal: 设置True运行在本地使用，设置FALSE 运行在flyai服务器上使用
    :param size: 每类的数据集扩容到size大小
    :param classify_count: 分类的数量
    :return: flyai的dataset类
    '''
    # step 0 : read csv
    # flyai_source = readCsv_onFlyai(readCsvOnLocal)
    flyai_source = SourceByWangyi().source_csv
    # step 1 : csv to dataframe
    dataframe_train = pd.DataFrame(data=flyai_source.data)
    dataframe_test = pd.DataFrame(data=flyai_source.val)
    # step 2 : extend csv(dataframe)
    dataframe_train = ExtendCsvToSize(dataframe_train, size=train_size, classify_count=classify_count)
    dataframe_test = ExtendCsvToSize(dataframe_test, size=val_size, classify_count=classify_count)
    # step 3 : save csv
    dataframe_train.to_csv(os.path.join(DATA_PATH, 'wangyi-train.csv'), index=False)
    dataframe_test.to_csv(os.path.join(DATA_PATH, 'wangyi-test.csv'), index=False)
    # step 4 : load to flyai.dataset
    dataset_extend_newone = Dataset(source=readCustomCsv("wangyi-train.csv", "wangyi-test.csv"))
    return dataset_extend_newone

def getDatasetListByClassfy(classify_count=3):
    test_source = SourceByWangyi()
    xx, yy = test_source.get_sliceCSVbyClassify(classify_count=classify_count)
    list_tmp=[]
    for epoch in range(classify_count):
        dataset = Dataset(source=readCustomCsv(xx[epoch], yy[epoch]))
        list_tmp.append(dataset)

    return list_tmp

if __name__=='__main__':

    dataset_slice = getDatasetListByClassfy(classify_count=3)
    x_train_slice,y_train_slice,x_val_slice, y_val_slice = [], [],[],[]
    for epoch in range(3):
        x_1,y_1,x_2,y_2 = dataset_slice[epoch].get_all_processor_data()
        # x_tmp, y_tmp = dataset_slice[epoch].get_all_validation_data()
        x_train_slice.append(x_1)
        y_train_slice.append(y_1)
        x_val_slice.append(x_2)
        y_val_slice.append(y_2)
        print('class-',epoch,'x_train.shape',x_1.shape)
        print('class-',epoch, 'y_train.shape', y_1.shape)
        print('class-', epoch, 'x_val.shape', x_2.shape)
        print('class-', epoch, 'y_val.shape', y_2.shape)

