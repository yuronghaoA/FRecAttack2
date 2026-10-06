import pickle
import numpy as np
from parse import args


def processing_data(uid_list, dataset):
    rating_max = -9999
    rating_min = 9999
    items = {}
    ratings = {}
    for uid in uid_list:
        items[uid] = []
        ratings[uid] = []
        rating = dataset[uid]
        for i in range(len(rating)):
            item, rate, _ = rating[i]
            items[uid].append(item)
            ratings[uid].append(int(rate))
        if len(ratings[uid]) > 0:
            rating_max = max(rating_max, max(ratings[uid]))
            rating_min = min(rating_min, min(ratings[uid]))

    return items, ratings, rating_max, rating_min


def social_data_loader():
    data_file = open('./Data/' + args.data + '.pkl', 'rb')
    [train_data, valid_data, test_data, user_id_list, item_id_list, social] = pickle.load(data_file)
    data_file.close()
    return train_data, valid_data, test_data, user_id_list, item_id_list, social


def load_file(file_path):
    m_item, all_pos = 0, []
    with open(file_path, "r") as f:
        for line in f.readlines():
            pos = list(map(int, line.rstrip().split(' ')))[1:]
            if pos:
                m_item = max(m_item, max(pos) + 1)
            all_pos.append(pos)
    return m_item, all_pos


def data_loader():
    data_path = "Data/"+args.data
    train_path = data_path + '/train.dat'
    valid_path = data_path + '/valid.dat'
    test_path = data_path + '/test.dat'

    m_item_train, traindata = load_file(train_path)
    m_item_valid, validdata = load_file(valid_path)
    m_item_test, testdata = load_file(test_path)

    m_item = max(m_item_test, m_item_train, m_item_valid)
    m_user = len(traindata)

    return m_item, m_user, traindata, validdata, testdata


def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)
