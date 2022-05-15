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
import streamlit as st


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
    #plt.show()
    with st.expander("Yıllık yangın Sayısı"):
        st.write("""
            İstanbul şehrinde 1998 ve 2017 yılları arasında gerçekleşen yangınların sayıları gösterilmektedir.
            Bu gösterimde, sadece ay ve toplam yangın sayısına odaklanılmıştır.
            Hangi yıllarda yangın sayısının artış gösterdiğine dair çıkarım bu grafiğe bakarak yapılabilir.
        """)
        st.pyplot(plt)



    # data gruplaştırma : year, state, month
    yearTableState = dataset.groupby(by=['year', 'state', 'month']).sum().reset_index()
    print(yearTableState)



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
    #plt.show()










    with st.expander("Aylık yangın Sayısı"):
        st.write("""
            istanbul'un tüm ilçeleri genelinde ve tüm yıllar için bir gösterim yapılmıştır.
            Ay bazlı bu gösterimde, tüm yıllar ve tüm ilçelerdeki veriler gruplanarak, İstanbul genelinde aylık bir yangın tahmin
            çıkarımının görselleştirilmesi hedeflenmiştir. Bu gösterim, gelecekteki yangın tahmininde hangi aylarda yangın önleme ve durdurma çalışmalarına
            ağırlık verilmesi gerektiği kararına etki edecektir.
        """)
        st.pyplot(plt)




    # İLÇELERE GÖRE OLAN YANGIN SAYISI GRİD BOX
    plt.figure(figsize=(30, 10))
    sns.boxplot(x='state', y='number', data=yearTableState, palette="autumn")
    plt.title('İLÇELERE GÖRE YANGINLAR', fontsize=25)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=15)
    #plt.show()
    with st.expander("İlçe bazlı yangın sayısı"):
        st.write("""
            İlçelerdeki yangınların gerçekleşme sayıları temel alınarak yapılan bu görselleştirme sayesinde, 
            gelecek seneler için hangi ilçelerde yangın önlem çalışmalarına ağırlık verilmesi gerektiği kararına 
            etki edecek değerler gösterilmiş olur.
        """)
        st.pyplot(plt)







