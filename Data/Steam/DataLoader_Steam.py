import pandas as pd
import csv
import random

user_ratio = 1
item_ratio = 1/3
item_limit = 5  #交互数最小取值
ratio_split_enable = False  # 划分方式，False为按个数划分，True为按比例划分
split_ratio = "6:1:1"  #划分比例，可以两个或者三个，划分方式关联使用


def Dataloader():
    file_name = "00_steam-200k.csv"
    file = pd.read_csv(file_name, names=['uid','iid','recording','rating','_'],usecols=['uid','iid','recording'])
    DF_data = pd.DataFrame(file)

    num_dic_user = {}
    for uid, group in DF_data.groupby("uid"):
        num_dic_user[uid] = len(group)
    uid_list = sorted(num_dic_user.items(), key=lambda x: x[1], reverse=True)[0:int(len(num_dic_user) * user_ratio)]
    total_uid_list = [x[0] for x in uid_list]

    num_dic_item = {}
    for iid, group in DF_data.groupby("iid"):
        num_dic_item[iid] = len(group)
    iid_list = sorted(num_dic_item.items(), key=lambda x: x[1], reverse=True)[0:int(len(num_dic_item) * item_ratio)]
    iid_list = [x[0] for x in iid_list]

    # 次筛
    true_uid_list = []
    true_iid_list = []
    for uid, group in DF_data.groupby('uid'):
        if uid in total_uid_list:
            num = 0
            item_list = []
            for item in group['iid'].tolist():
                if (item in iid_list) and (item not in item_list):
                    num += 1
                    item_list.append(item)
            if num >= item_limit:
                true_uid_list.append(uid)
                true_iid_list.extend(item_list)
    iid_list = list(set(true_iid_list))

    dataset = {}
    for uid, group in DF_data.groupby("uid"):
        if uid in true_uid_list:
            item_ = []
            for item in group['iid']:
                if item in iid_list:
                    item_.append(iid_list.index(item))
            dataset[true_uid_list.index(uid)] = list(set(item_))

    ratio_list = split_ratio.split(":")
    if ratio_split_enable:
        ratio_list = list(map(float,ratio_list))
        with open("01_train.dat", 'w') as f1:
            for key, items in dataset.items():
                f1.write(str(key) + ' ')
                train_num = int(ratio_list[0]*len(items)/float(sum(ratio_list)))
                for item in items[:train_num]:
                    f1.write(str(item) + ' ')
                f1.write('\n')
        with open("02_test.dat", 'w') as f1:
            for key, items in dataset.items():
                f1.write(str(key) + ' ')
                train_num = int(ratio_list[0] * len(items)/float(sum(ratio_list)))
                test_num  = int(ratio_list[1] * len(items)/float(sum(ratio_list)))
                for item in items[train_num:train_num+test_num]:
                    f1.write(str(item) + ' ')
                f1.write('\n')
        if len(ratio_list) == 3:
            with open("03_valid.dat", 'w') as f1:
                for key, items in dataset.items():
                    f1.write(str(key) + ' ')
                    train_num = int(ratio_list[0] * len(items)/float(sum(ratio_list)))
                    test_num  = int(ratio_list[1] * len(items)/float(sum(ratio_list)))
                    valid_num = int(ratio_list[2] * len(items)/float(sum(ratio_list)))
                    for item in items[train_num+test_num:train_num + test_num+valid_num]:
                        f1.write(str(item) + ' ')
                    f1.write('\n')
    else:
        ratio_list = list(map(int, ratio_list))
        with open("01_train.dat", 'w') as f1:
            for key, items in dataset.items():
                f1.write(str(key) + ' ')
                for item in items[:-sum(ratio_list[1:])]:
                    f1.write(str(item) + ' ')
                f1.write('\n')
        if len(ratio_list)==2:
            with open("02_test.dat", 'w') as f1:
                for key, items in dataset.items():
                    f1.write(str(key) + ' ')
                    for item in items[-sum(ratio_list[1:]):]:
                        f1.write(str(item) + ' ')
                    f1.write('\n')
        else:
            with open("02_test.dat", 'w') as f1:
                for key, items in dataset.items():
                    f1.write(str(key) + ' ')
                    for item in items[-sum(ratio_list[1:]):-ratio_list[-1]]:
                        f1.write(str(item) + ' ')
                    f1.write('\n')
            with open("03_valid.dat", 'w') as f1:
                for key, items in dataset.items():
                    f1.write(str(key) + ' ')
                    for item in items[-ratio_list[-1]:]:
                        f1.write(str(item) + ' ')
                    f1.write('\n')



if __name__ == "__main__":
    Dataloader()