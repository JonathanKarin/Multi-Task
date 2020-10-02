from one_hot import *
from prepare_data import *
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Input, Conv1D, MaxPooling2D, MaxPooling1D, \
    AveragePooling1D, LSTM, Dropout, Bidirectional, LeakyReLU
from keras.layers.merge import concatenate
from scipy.stats import pearsonr
from keras.regularizers import l2


def main():
    print("main 4")
    x_train, x_test, y_train, y_test, number_of_SETB, list_of_nans, colums4 , params_dict = get_data75()
    X = Input(shape=(75, 4))
    conv_kernel_long = Conv1D(params_dict["filters_long"], kernel_size=params_dict["filters_long_length"], activation='relu', use_bias=True,
                              kernel_regularizer=l2(params_dict["regu"]))(X)  # Long kernel - its purpose is to identify structure preferences
    conv_kernel_11 = Conv1D(filters=params_dict["filters1"], kernel_size=11, activation='relu', use_bias=True,
                            kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 11 nucleotides
    conv_kernel_9 = Conv1D(filters=params_dict["filters1"], kernel_size=9, activation='relu', use_bias=True,
                           kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 9 nucleotides
    conv_kernel_7 = Conv1D(filters=params_dict["filters1"], kernel_size=7, activation='relu', use_bias=True,
                           kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 7 nucleotides
    conv_kernel_5 = Conv1D(filters=params_dict["filters1"], kernel_size=5, activation='relu', use_bias=True,
                           kernel_regularizer=l2(params_dict["regu"]))(X)  # kernel of 5 nucleotides
    conv_kernel_5_sec = Conv1D(filters=params_dict["filters_sec"], kernel_size=5, activation='relu', use_bias=True,
                             kernel_regularizer=l2(params_dict["regu"]))(X) # kernel of 5 nucleotides - second path

    max_pool_long = MaxPooling1D(pool_size=(74 - params_dict["filters_long_length"]))(conv_kernel_long)
    max_pool_11 = MaxPooling1D(pool_size=(65))(conv_kernel_11)
    max_pool_9 = MaxPooling1D(pool_size=(67))(conv_kernel_9)
    max_pool_7 = MaxPooling1D(pool_size=(69))(conv_kernel_7)
    max_pool_5 = MaxPooling1D(pool_size=(71))(conv_kernel_5)
    max_pool_5_sec = MaxPooling1D(pool_size=(71))(conv_kernel_5_sec)
    merge2 = concatenate([max_pool_11, max_pool_7, max_pool_long, max_pool_9, max_pool_5]) #merge first path
    fl_rel = Flatten()(merge2) #Flatten layer
    fl_sec = Flatten()(max_pool_5_sec) #Flatten layer - second path
    drop_fl_sec = Dropout(params_dict["dropout"], name="drop_fl_el")(fl_sec) #Dropout
    drop_flat = Dropout(params_dict["dropout"], name="drop_flat")(fl_rel)
    hidden_dense_sec = Dense(params_dict["hidden_sec"], activation='relu')(drop_fl_sec)
    hidden_dense_relu = Dense(params_dict["hidden1"], activation='relu')(drop_flat)  # 4096
    drop_hidden_dense_relu = Dropout(params_dict["dropout"], name="drop_hidden_dense_relu")(hidden_dense_relu)
    hidden_dense_relu1 = Dense(params_dict["hidden2"], activation='relu')(drop_hidden_dense_relu)  # 1024 best
    merge_4 = concatenate([hidden_dense_sec, hidden_dense_relu1, drop_flat, hidden_dense_relu])
    Y_1 = Dense(244)(merge_4)
    Y = LeakyReLU(alpha=params_dict["leaky_alpha"])(Y_1)
    model_func = Model(inputs=X, outputs=Y)
    model_func.compile(loss='logcosh', optimizer='adam')  # adam
    print(model_func.summary())
    model_func.fit(x_train, y_train, batch_size=params_dict["batch"], epochs=params_dict["epochs"], verbose=1)  # 4096 , 92 best result
    model_pred = model_func.predict(x_test)
    y_test_reshaped = (y_test.reshape(number_of_SETB, 244)).T
    model_pred_reshaped = (model_pred.reshape(number_of_SETB, 244)).T
    nan_num = 0
    list_of_res = []
    for i in range(244):
        model_pred_without_nans = np.delete(model_pred_reshaped[i][:], list_of_nans[i])
        model_pred_reshaped_1d = model_pred_without_nans
        y_test_without_nans = np.delete(y_test_reshaped[i][:], list_of_nans[i])
        y_test_reshaped_1d = y_test_without_nans
        res = pearsonr(y_test_reshaped_1d, model_pred_reshaped_1d)
        print(colums4[i] + " " + str(res[0]))
        list_of_res.append(float(res[0]))
    import statistics
    print("Mean:" + str(statistics.mean(list_of_res)))
    print("std:" + str(statistics.stdev(list_of_res)))
    print("Median:" + str(statistics.median(list_of_res)))
    print("Max:" + str(max(list_of_res)))
    print("Min:" + str(min(list_of_res)))
    print(str(nan_num) + " Is the number of nan's value")
    model_func.save("model_41_9.h5")
    pd.DataFrame(list_of_res).to_csv("res.csv")


if __name__ == "__main__":
    main()

