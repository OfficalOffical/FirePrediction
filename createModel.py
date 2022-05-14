import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OrdinalEncoder as Encoder
from sklearn.model_selection import train_test_split as tts
from keras.layers import Conv1D, Flatten
from sklearn.preprocessing import MinMaxScaler as Scaler



import matplotlib.pyplot as plt
import numpy as np

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


def create(dataset):

    dataset.drop(['date'], axis=1, inplace=True)
    print(dataset)

    dataset['number'] = dataset['number'].astype('int')
    print('10 YILDA KAYDEDİLEN TOPLAM YANGIN SAYISI :', dataset['number'].sum())

    yearTable = pd.pivot_table(dataset, values="number", index=["year"], aggfunc=np.sum)
    print(yearTable)

    ax.xaxis.set_major_locator(plt.MaxNLocator(19))
    ax.set_xlim(1998, 2017)

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))


    # data gruplaştırma : year, state, month
    yearTableState = dataset.groupby(by=['year', 'state', 'month']).sum().reset_index()
    print(yearTableState)

    # Group data by year, state, month
    yearTableState = dataset.groupby(by=['year', 'state', 'month']).sum().reset_index()
    print(yearTableState)

    # #Figure size
    plt.figure(figsize=(18, 10))

    # The plot
    sns.boxplot(x='month',
                order=['OCAK', 'SUBAT', 'MART', 'NISAN', 'MAYIS', 'HAZIRAN', 'TEMMUZ', 'AGUSTOS', 'EYLUL', 'EKIM',
                       'KASIM', 'ARALIK'],
                y='number', data=yearTableState, palette="autumn", saturation=1, width=0.9, fliersize=4, linewidth=2)

    plt.title('İSTANBUL AYLIK YANGIN SAYILARI', fontsize=25)
    plt.xlabel('AYLAR', fontsize=20)
    plt.ylabel('YANGIN SAYISI', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)


    # checking monthwise trend
    plt.figure(figsize=(28, 10))
    sns.boxplot(x='state', y='number', data=yearTableState)
    plt.title('İLÇELERE GÖRE YANGIN SAYILARI', fontsize=25)
    plt.show()
