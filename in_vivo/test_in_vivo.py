from common import one_hot
from common import prepare_data
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from keras.models import load_model


def main():
    model_func = load_model('model_75_4.h5')
    import statistics
    df = pd.read_csv('eclip.csv')
    df['res'] = 0
    print("r")
    list_of_res = []
    for index, row in df.iterrows():
        x, y = prepare_data.read_eclip_75("eclip/" + row["ENCODE"] + ".bed.ext.fa", "eclip/" + row["ENCODE"] + ".bed.control.ext.fa")
        model_pred_no_lstm = model_func.predict(x)
        model_pred_no_lstm = model_pred_no_lstm.T
        auc = roc_auc_score(y, model_pred_no_lstm[row["pred_index"]])
        print(row["ENCODE"] + " " + str(auc) + " :auc ")
        list_of_res.append(float(auc))
    pd.DataFrame(list_of_res).to_csv("auc_eclip75_new.csv")
    print("Mean:" + str(statistics.mean(list_of_res)))
    print("std:" + str(statistics.stdev(list_of_res)))
    print("Median:" + str(statistics.median(list_of_res)))
    print("Max:" + str(max(list_of_res)))
    print("Min:" + str(min(list_of_res)))


if __name__ == "__main__":
    main()

