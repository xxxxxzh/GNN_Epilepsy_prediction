import numpy as np
import pandas as pd
import torch
import os

def load_features(feat_path, dtype=np.float32):
    '''

    :param feat_path:
    :param dtype:
    :return: feat (time_len, num_node)
    '''
    # print(feat_path)
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat

def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj

def generate_sequence_data(feature,label:int,seq_len:int):
    '''

    :param feature: shape -> (num_feature,num_node)
    :param label:   choice (0,1)
    :param seq_len:
    :return: data_x: (batch_size, num_node, seq_len), data_y: (batch_size)
    '''
    tot_len = feature.shape[0]
    new_feature = [feature[i * seq_len : (i + 1) * seq_len,:] for i in range(tot_len // seq_len)]
    # new_feature = [feature[i : i + seq_len, :] for i in range(tot_len - seq_len + 1)]

    #(batch_size, seq_len, num_node)
    # print(np.array(new_feature).transpose([0,2,1]).shape)
    data_x = np.array(new_feature).transpose([0,2,1])
    # data_x = np.concatenate(new_feature,axis = 0).transpose([0,2,1])
    data_y = np.ones((data_x.shape[0],))
    data_y[:] = label
    return data_x, data_y

def multifiles_generate_sequence_data(filepath,label,seq_len,dtype = np.float32):
    '''

    :param filepath:
    :param seq_len:
    :param dtype:
    :return: data_x: (batch_size, num_node, seq_len), data_y: (batch_size)
    '''
    files = os.listdir(filepath)
    data_x, data_y = (None,None)
    for index, filename in enumerate(files):
        if (index == 0):
            file = os.path.join(filepath,filename)
            data = load_features(file,dtype)
            data_x, data_y = generate_sequence_data(data,label,seq_len)
        else:
            file = os.path.join(filepath, filename)
            data = load_features(file, dtype)
            tmp_x, tmp_y = generate_sequence_data(data, label, seq_len)

            data_x = np.concatenate([data_x,tmp_x],axis = 0)
            data_y = np.concatenate([data_y,tmp_y],axis = 0)
    return data_x, data_y


def multifiles_generate_dataset(ictal_path,preictal,seq_len, normalize = True,dtype=np.float32):
    '''

    :param ictal_path:  positive_path
    :param preictal:    negetive_path
    :param seq_len:     len(feature_dim)
    :param normalize:
    :param dtype:       data_type
    :return:  data_x: (batch_size, num_node, seq_len), data_y: (batch_size)
    '''

    # data_x, data_y = (None, None)
    data_x, data_y = multifiles_generate_sequence_data(ictal_path,1,seq_len,dtype)
    # if (data_x,data_y) == (None,None):
    #     data_x, data_y = multifiles_generate_sequence_data(preictal,0,seq_len,dtype)
    # else:
    tmp_x, tmp_y = multifiles_generate_sequence_data(preictal,0,seq_len,dtype)
    data_x = np.concatenate([data_x, tmp_x], axis = 0)
    data_y = np.concatenate([data_y, tmp_y], axis = 0)

    if normalize == True:
        max_val = np.max(data_x)
        data_x = data_x / max_val

    return data_x, data_y

def generate_torch_datasets(train_path, test_path, seq_len, normalize = True):

    '''
    :param train_path:
    :param test_path:
    :param seq_len:
    :param normorlize:
    :return: train_dataset, test_dataset
    '''
    train_ictal = os.path.join(train_path,'ictal')
    train_preictal = os.path.join(train_path, 'preictal')


    test_ictal = os.path.join(test_path, 'ictal')
    test_preictal = os.path.join(test_path, 'preictal')

    train_x, train_y = multifiles_generate_dataset(train_ictal, train_preictal, seq_len, normalize = normalize)
    test_x, test_y = multifiles_generate_dataset(test_ictal, test_preictal, seq_len, normalize = normalize)

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_x), torch.FloatTensor(test_y)
    )
    return train_dataset, test_dataset

'''
def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)
'''

'''
def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
'''