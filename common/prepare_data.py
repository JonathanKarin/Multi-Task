from one_hot import *
import numpy as np
from collections import Counter
import pandas as pd


def second_struct(seq):
    print(seq)
    ans = np.zeros((75, 5))
    f = open('struct/1.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[1:] == seq:
            for k in range(5):
                next_line = lines[i + 1 + k].split()
                for j in range(len(next_line)):
                    ans[j][k] = next_line[j]
            return ans
        if line[0] == '>':
            i += 6

    f = open('struct/2.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[1:] == seq:
            for k in range(5):
                next_line = lines[i + 1 + k].split()
                for j in range(len(next_line)):
                    ans[j][k] = next_line[j]
            return ans
        if line[0] == '>':
            i += 6
    f = open('struct/3.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[1:] == seq:
            for k in range(5):
                next_line = lines[i + 1 + k].split()
                for j in range(len(next_line)):
                    ans[j][k] = next_line[j]
            return ans
        if line[0] == '>':
            i += 6
    f = open('struct/4.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[1:] == seq:
            for k in range(5):
                next_line = lines[i + 1 + k].split()
                for j in range(len(next_line)):
                    ans[j][k] = next_line[j]
            return ans
        if line[0] == '>':
            i += 6
    f = open('struct/5.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[1:] == seq:
            for k in range(5):
                next_line = lines[i + 1 + k].split()
                for j in range(len(next_line)):
                    ans[j][k] = next_line[j]
            return ans
        if line[0] == '>':
            i += 6
    f = open('struct/6.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[1:] == seq:
            for k in range(5):
                next_line = lines[i + 1 + k].split()
                for j in range(len(next_line)):
                    ans[j][k] = next_line[j]
            return ans
        if line[0] == '>':
            i += 6
    f = open('struct/7.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line[1:] == seq:
            for k in range(5):
                next_line = lines[i + 1 + k].split()
                for j in range(len(next_line)):
                    ans[j][k] = next_line[j]
            return ans
        if line[0] == '>':
            i += 6
    print("not found!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("not found!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("not found!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("not found!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("not found!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("not found!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("not found!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    return 0


def read_clip(path_pos, path_neg):
    pos_list = []
    f_pos = open(path_pos)
    for line in f_pos:
        if (line[0] == '>'):
            continue
        elif (line.find("t") == -1) & (line.find("T") == -1) & (
                (line[0] == 'a') | (line[0] == 'c') | (line[0] == 'g') | (line[0] == 'u')):
            pos_list.append(re.sub('\n', '', (re.sub('[a-z]', '', line))))
    neg_list = []
    f_neg = open(path_neg)
    for line in f_neg:
        if (line[0] == '>'):
            continue
        elif (line.find("t") == -1) & (line.find("T") == -1) & (
                (line[0] == 'a') | (line[0] == 'c') | (line[0] == 'g') | (line[0] == 'u')):
            neg_list.append(re.sub('\n', '', (re.sub('[a-z]', '', line))))

    x = np.zeros((len(neg_list) + len(pos_list), 75, 4))
    y = np.zeros((len(neg_list) + len(pos_list)))
    i = 0
    for line in pos_list:
        x[i] = one_hot_encode_75(line)
        y[i] = 1
        i = i + 1

    for line in neg_list:
        x[i] = one_hot_encode_75(line)
        i = i + 1
    return x, y


def get_data1():
    data_path = "../data/norm_data.txt"

    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums4 = columns[3:]
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    x_shaped_a = np.zeros((120326, 75, 4))  # 120326
    x_shaped_b = np.zeros((121031, 75, 4))  # 121031
    y_shaped_a = np.zeros((120326, 244))  # 120326
    y_shaped_b = np.zeros((121031, 244))  # 121031
    x_struct_a = np.zeros((120326, 75, 5))  # 120326
    x_struct_b = np.zeros((121031, 75, 5))  # 120326

    i = 0
    for index, row in Train.iterrows():
        x_shaped_a[i] = one_hot_encode_75(row[0])  # encode_me(seq)
        x_struct_a[i] = second_struct(row[0])
        y_shaped_a[i] = row[1:]
        i = i + 1

    i = 0
    for index, row in Test.iterrows():
        x_shaped_b[i] = one_hot_encode_75(row[0])  # encode_me(seq)
        x_struct_b[i] = second_struct(row[0])
        y_shaped_b[i] = row[1:]
        i = i + 1

    return x_shaped_a, x_shaped_b, y_shaped_a, y_shaped_b, 121031, list_of_nans, colums4, x_struct_a, x_struct_b


def get_data41x4():
    data_path = "../data/norm_data.txt"

    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums4 = columns[3:]
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    x_shaped_a = np.zeros((120326, 41, 4))  # 120326
    x_shaped_b = np.zeros((121031, 41, 4))  # 121031
    y_shaped_a = np.zeros((120326, 244))  # 120326
    y_shaped_b = np.zeros((121031, 244))  # 121031

    i = 0
    for index, row in Train.iterrows():
        x_shaped_a[i] = one_hot_encode_(row[0])  # encode_me(seq)
        y_shaped_a[i] = row[1:]
        i = i + 1

    i = 0
    for index, row in Test.iterrows():
        x_shaped_b[i] = one_hot_encode_(row[0])  # encode_me(seq)
        y_shaped_b[i] = row[1:]
        i = i + 1

    return x_shaped_a, x_shaped_b, y_shaped_a, y_shaped_b, 121031, list_of_nans, colums4


def find_nans(col_of_data):
    bool_col = col_of_data.astype(float).isna()
    list_nan_col = np.where(bool_col)
    return list_nan_col


def get_data():
    data_path = "../data/norm_data.txt"
    # f = open(data_path, "r")
    # word_counts = Counter(f.read().split())
    # set_a_size = word_counts.get('SetA')
    # set_b_size = word_counts.get('SetB')
    # f.close()
    f = open(data_path, "r")
    i = 0
    j = 0
    k = 0
    x_shaped_a = np.zeros((120326, 41, 4))
    x_shaped_b = np.zeros((121031, 41, 4))
    y_shaped_a = np.zeros((120326, 1, 244))
    y_shaped_b = np.zeros((121031, 1, 244))
    print("Start Encoding")
    #   print(str(set_b_size) + " is the size of SetB")
    #  print(str(set_a_size) + " is the size of SetA")
    for line in f:
        if i > 0:
            item_ = line.split()
            if item_.__getitem__(0) == 'SetA':
                seq = item_.__getitem__(1)
                x_shaped_a[j, :, :] = one_hot_encode_(seq)  # encode_me(seq)
                y_shaped_a[j, :, :] = encode_y(line)
                j = j + 1
            else:
                seq = item_.__getitem__(1)
                x_shaped_b[k, :, :] = one_hot_encode_(seq)  # encode_me(seq)
                y_shaped_b[k, :, :] = encode_y(line)
                k = k + 1

        else:
            labels = line.split()
            del labels[0:3]
        i = i + 1
        # print(line)
        # print("iteration number:" +str(i))
    return x_shaped_a, x_shaped_b, y_shaped_a, y_shaped_b, k


def encode_y(line):
    line = line.replace("NaN", "0")
    next_row = line.split()
    del next_row[0:3]
    return next_row



def get_data_saved():
    data_path = "../data/norm_data.txt"

    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums4 = columns[3:]
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    x_shaped_a = np.load('x_shaped_a.npy')  # load
    x_shaped_b = np.load('x_shaped_b.npy')  # load
    y_shaped_a = np.load('y_shaped_a.npy')  # load
    y_shaped_b = np.load('y_shaped_b.npy')  # load
    x_struct_a = np.load('x_struct_a.npy')  # load
    x_struct_b = np.load('x_struct_b.npy')  # load
    return x_shaped_a, x_shaped_b, y_shaped_a, y_shaped_b, 121031, list_of_nans, colums4, x_struct_a, x_struct_b


def save_data_5():
    data_path = "norm_data.txt"

    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums1 = list(df)
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    # for i in columns:
    #   print(df[i].describe())
    x_shaped_a = np.zeros((120326, 75, 9))  # 120326
    x_shaped_b = np.zeros((121031, 75, 9))  # 121031
    y_shaped_a = np.zeros((120326, 244))  # 120326
    y_shaped_b = np.zeros((121031, 244))  # 121031
    f = open('all_seq.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    for index, row in Train.iterrows():
        print(str(i))
        con = 0
        x_shaped_a[i, :, 0:4] = one_hot_encode_75(row[0])  # encode_me(seq)
        seq = row[0]
        ans = np.zeros((75, 5))
        p = 0
        while (p < len(lines)) & (con == 0):
            line = lines[p]
            if line[1:] == seq:
                con = 1
                for k in range(5):
                    next_line = lines[p + 1 + k].split()
                    for j in range(len(next_line)):
                        ans[j][k] = next_line[j]
            if line[0] == '>':
                p += 6
        x_shaped_a[i, :, 4:9] = ans
        y_shaped_a[i] = row[1:]
        i = i + 1
    i = 0
    for index, row in Test.iterrows():
        print(str(i))
        x_shaped_b[i, :, 0:4] = one_hot_encode_75(row[0])  # encode_me(seq)
        seq = row[0]
        ans = np.zeros((75, 5))
        p = 0
        while p < len(lines):
            line = lines[p]
            if line[1:] == seq:
                for k in range(5):
                    next_line = lines[p + 1 + k].split()
                    for j in range(len(next_line)):
                        ans[j][k] = next_line[j]
            if line[0] == '>':
                p += 6
        x_shaped_b[i, :, 4:9] = ans
        y_shaped_b[i] = row[1:]
        i = i + 1
    np.save('x_shaped_a_75x9.npy', x_shaped_a)
    np.save('x_shaped_b_75x9.npy', x_shaped_b)
    return


def get_data_saved_75x9():
    data_path = "norm_data.txt"

    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums4 = columns[3:]
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    x_shaped_a = np.load('x_shaped_a_75x9.npy')  # load
    x_shaped_b = np.load('x_shaped_b_75x9.npy')  # load
    y_shaped_a = np.load('y_shaped_a.npy')  # load
    y_shaped_b = np.load('y_shaped_b.npy')  # load
    return x_shaped_a, x_shaped_b, y_shaped_a, y_shaped_b, 121031, list_of_nans, colums4


def save_data_41():
    data_path = "norm_data.txt"
    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums1 = list(df)
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    # for i in columns:
    #   print(df[i].describe())
    x_shaped_a = np.zeros((120326, 41, 9))  # 120326
    x_shaped_b = np.zeros((121031, 41, 9))  # 121031
    y_shaped_a = np.zeros((120326, 244))  # 120326
    y_shaped_b = np.zeros((121031, 244))  # 121031
    f = open('all_seq.txt', 'r+')
    lines = f.read().splitlines()
    i = 0
    for index, row in Train.iterrows():
        print(str(i))
        con = 0
        x_shaped_a[i, :, 0:4] = one_hot_encode_(row[0])  # encode_me(seq)
        seq = row[0]
        ans = np.zeros((41, 5))
        p = 0
        while (p < len(lines)) & (con == 0):
            line = lines[p]
            if line[1:] == seq:
                con = 1
                for k in range(5):
                    next_line = lines[p + 1 + k].split()
                    for j in range(len(next_line)):
                        ans[j][k] = next_line[j]
            if line[0] == '>':
                p += 6
        x_shaped_a[i, :, 4:9] = ans
        y_shaped_a[i] = row[1:]
        i = i + 1
    i = 0
    for index, row in Test.iterrows():
        print(str(i))
        x_shaped_b[i, :, 0:4] = one_hot_encode_(row[0])  # encode_me(seq)
        seq = row[0]
        ans = np.zeros((41, 5))
        p = 0
        while p < len(lines):
            line = lines[p]
            if line[1:] == seq:
                for k in range(5):
                    next_line = lines[p + 1 + k].split()
                    for j in range(len(next_line)):
                        ans[j][k] = next_line[j]
            if line[0] == '>':
                p += 6
        x_shaped_b[i, :, 4:9] = ans
        y_shaped_b[i] = row[1:]
        i = i + 1
    np.save('x_shaped_a_41x9.npy', x_shaped_a)
    np.save('x_shaped_b_41x9.npy', x_shaped_b)
    return


def get_data_saved_41x9():
    data_path = "../data/norm_data.txt"

    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums4 = columns[3:]
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    x_shaped_a = np.load('x_shaped_a_41x9.npy')  # load
    x_shaped_b = np.load('x_shaped_b_41x9.npy')  # load
    y_shaped_a = np.load('y_shaped_a.npy')  # load
    y_shaped_b = np.load('y_shaped_b.npy')  # load
    params_dict = {
        "dropout": 0.382233801349954,
        "epochs": 78,
        "batch" : 4096,
        "regu": 5.6215002041656515e-06,
        "hidden1" : 6029,
        "hidden2" : 1168,
        "filters1" : 2376,
        "hidden_sec" : 152,
        "filters_sec" : 151,
        "leaky_alpha" : 0.23149394545024274,
        "filters_long_length" : 24,
        "filters_long" : 51
    }

    return x_shaped_a, x_shaped_b, y_shaped_a, y_shaped_b, 121031, list_of_nans, colums4 , params_dict


def get_data75():
    data_path = "../data/norm_data.txt"

    df = pd.read_csv(data_path, delimiter="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row

    df.columns = new_header  # set the header row as the df header
    columns = list(df)
    colums4 = columns[3:]
    i = 0
    list_of_nans = []
    pre_test_for_nans = df[df['Probe_Set'] == 'SetB']
    for j in columns:
        if (i > 2):
            # print(str(df[j].astype(float).mean()))
            list_of_nans.append(find_nans(pre_test_for_nans[j]))
            df[j] = df[j].astype(float).fillna(df[j].astype(float).mean())
            df[j] = df[j].clip(lower=-100, upper=df[j].quantile(0.995))
        i = i + 1
    Train = df[df['Probe_Set'] == 'SetA']
    Train = Train.drop(['Probe_Set', 'Probe_ID'], axis=1)
    Test = df[df['Probe_Set'] == 'SetB']
    Test = Test.drop(['Probe_Set', 'Probe_ID'], axis=1)
    columns = list(df)
    x_shaped_a = np.zeros((120326, 75, 4))  # 120326
    x_shaped_b = np.zeros((121031, 75, 4))  # 121031
    y_shaped_a = np.zeros((120326, 244))  # 120326
    y_shaped_b = np.zeros((121031, 244))  # 121031

    i = 0
    for index, row in Train.iterrows():
        x_shaped_a[i] = one_hot_encode_75(row[0])  # encode_me(seq)
        y_shaped_a[i] = row[1:]
        i = i + 1

    i = 0
    for index, row in Test.iterrows():
        x_shaped_b[i] = one_hot_encode_75(row[0])  # encode_me(seq)
        y_shaped_b[i] = row[1:]
        i = i + 1
    params_dict = {
        "dropout": 0.382233801349954,
        "epochs": 78,
        "batch" : 4096,
        "regu": 5.6215002041656515e-06,
        "hidden1" : 6029,
        "hidden2" : 1168,
        "filters1" : 2376,
        "hidden_sec" : 152,
        "filters_sec" : 151,
        "leaky_alpha" : 0.23149394545024274,
        "filters_long_length" : 24,
        "filters_long" : 51
    }

    return x_shaped_a, x_shaped_b, y_shaped_a, y_shaped_b, 121031, list_of_nans, colums4 ,params_dict

