import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OrdinalEncoder as Encoder
from sklearn.model_selection import train_test_split as tts
from keras.layers import Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler as Scaler
#from keras.saving.saved_model.json_utils import Encoder
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axis as ax
import numpy as np
from IPython import get_ipython
from IPython import get_ipython
import seaborn as sns


def create(dataset):

    dataset.drop(['date'], axis=1, inplace=True)
    print(dataset)


    dataset['number'] = dataset['number'].astype('int')
    print('10 YILDA KAYDEDİLEN TOPLAM YANGIN SAYISI :', dataset['number'].sum())
    # YILLAR VE TOPLAM YANGINLAR TABLOSU
    yearTable = pd.pivot_table(dataset, values="number", index=["year"], aggfunc=np.sum)
    print(yearTable)
    print("----------------------------------------------------------------------------------")


    # YILLARA GÖRE OLAN YANGIN SAYISI GRİD BAR
    plt.figure(figsize=(30, 10))
    # plot
    ax = sns.boxplot(x='year', y='number', data=dataset, palette="autumn")
    plt.title("YILLARA GÖRE İSTANBULDA GERÇEKLEŞEN YANGINLAR : 1998 - 2017", fontsize=25)
    plt.xlabel("YIL", fontsize=10)
    plt.ylabel("YANGIN SAYISI", fontsize=10)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()


    # data gruplaştırma : year, state, month
    yearTableState = dataset.groupby(by=['year', 'state', 'month']).sum().reset_index()
    print(yearTableState)
    print("----------------------------------------------------------------------------------")


    # AYLARA GÖRE OLAN YANGIN SAYISI GRİD BOX
    plt.figure(figsize=(30, 10))

    order_list = ['OCAK', 'SUBAT', 'MART', 'NISAN', 'MAYIS', 'HAZIRAN', 'TEMMUZ', 'AGUSTOS', 'EYLUL', 'EKIM', 'KASIM', 'ARALIK']
    # The plot
    sns.boxplot(x='month', order=order_list,
                y='number', data=yearTableState, palette="autumn", saturation=1, width=0.9, fliersize=4, linewidth=2)

    plt.title('İSTANBUL AYLIK YANGIN SAYILARI', fontsize=25)
    plt.xlabel('AYLAR', fontsize=20)
    plt.ylabel('YANGIN SAYISI', fontsize=20)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=15)
    plt.show()
    print("----------------------------------------------------------------------------------")


    # İLÇELERE GÖRE OLAN YANGIN SAYISI GRİD BOX
    plt.figure(figsize=(30, 10))
    sns.boxplot(x='state', y='number', data=yearTableState, palette="autumn")
    plt.title('İLÇELERE GÖRE YANGIN SAYILARI', fontsize=25)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=15)
    plt.show()
    print("----------------------------------------------------------------------------------")
