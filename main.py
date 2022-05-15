import pandas as pd








from createModel import *
from base64Temp import mainBase





dataset = pd.read_csv('amazon.csv', encoding='unicode_escape', sep=';')

create(dataset)

features = dataset[['state', 'month']]
targets = dataset['number']




x_train, x_test, y_train, y_test = tts(features, targets, test_size=0.2,
                                       random_state=42)

encoder_state = Encoder()
encoder_month = Encoder()

x_train['state'] = encoder_state.fit_transform(x_train['state'].values.reshape(-1, 1))
x_train['month'] = encoder_month.fit_transform(x_train['month'].values.reshape(-1, 1))

x_test['state'] = encoder_state.fit_transform(x_test['state'].values.reshape(-1, 1))
x_test['month'] = encoder_month.fit_transform(x_test['month'].values.reshape(-1, 1))  # String to Int

# encoder_state.inverse_transform(x_train['state'].values.reshape(-1, 1))

# encoder_state.categories_


ss = Scaler()


y_train = ss.fit_transform(y_train.values.reshape(-1, 1))
y_test = ss.transform(y_test.values.reshape(-1, 1))



x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Input kısmı

model = Sequential([
    keras.Input(shape=(2, 1)),
    Conv1D(64, 2),

    Dense(48),
    Conv1D(32, 1),

    Flatten(),
    Dense(1)

])
model.summary()

from keras.optimizers import adam_v2

opt = adam_v2.Adam(learning_rate=0.0001)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=opt,
              metrics=['mse', 'mae'])
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),
          verbose=2)

evScore = model.evaluate(x_test, y_test)

predictions = model.predict(x_test)


encoder_state.inverse_transform(x_test[0].reshape(-1, 1))

tempSS = ss.inverse_transform(predictions)


x_train = x_train.squeeze(axis= 2)
x_test = x_test.squeeze(axis= 2)


y_train = ss.inverse_transform(y_train.reshape(-1, 1))
y_test = ss.inverse_transform(y_test.reshape(-1, 1))

xTestState = encoder_state.inverse_transform(x_test[:,0].reshape(-1, 1))
xTestMonth = encoder_month.inverse_transform(x_test[:,1].reshape(-1, 1))

print(xTestMonth.shape)
print(xTestState.shape)
print(tempSS.shape)


temp = pd.DataFrame({'xTestMonth': xTestMonth[:,0], 'xTestState':xTestState[:,0], 'tempSS': tempSS[:,0]})
tempSum = temp.groupby(by=['xTestMonth','xTestState','tempSS']).sum().reset_index()




tempSum['tempSS'] = tempSum['tempSS']/(tempSum['tempSS'].sum())*100

tempMean = tempSum['tempSS'].mean()
tempMax = tempSum['tempSS'].max()


with st.sidebar:
    st.metric(label="En yüksek yangın ihtimalli şehir ve ay", value=tempMax, delta=tempMax- tempMean)


def tempSumMonthGrid(tempSum):
    plt.figure(figsize=(30, 10))

    order_list = ['OCAK', 'SUBAT', 'MART', 'NISAN', 'MAYIS', 'HAZIRAN', 'TEMMUZ', 'AGUSTOS', 'EYLUL', 'EKIM', 'KASIM', 'ARALIK']
    # The plot
    sns.boxplot(x='xTestMonth', order=order_list,
                y='tempSS', data=tempSum, palette="autumn", saturation=1, width=0.9, fliersize=4, linewidth=2)

    plt.title('AYLARA GÖRE BEKLENEN YANGIN ORANLARI', fontsize=25)
    plt.xlabel('AYLAR', fontsize=20)
    plt.ylabel('YANGIN ORANI', fontsize=20)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=15)
    #plt.show()
    with st.expander("Aylara göre beklenen yangın tahmin oranları"):
        st.write("""
            Aylar içerisinde gerçekleşme olasılığı taşıyan yangınların model tarafından hesaplanan beklenme değerleri görselleştirilmiştir.
        """)
        st.pyplot(plt)





def tempSumStateGrid(tempSum):
    plt.figure(figsize=(30, 10))

    order_list = ['ARNAVUTKOY', 'ATASEHIR', 'BEYKOZ', 'BEYOGLU', 'CATALCA', 'ESENYURT', 'FATIH', 'KADIKOY', 'KAGITHANE', 'UMRANIYE', 'USKUDAR', 'ZEYTINBURNU']
    # The plot
    sns.boxplot(x='xTestState', order=order_list,
                y='tempSS', data=tempSum, palette="autumn", saturation=1, width=0.9, fliersize=4, linewidth=2)

    plt.title('İLÇELERE GÖRE BEKLENEN YANGIN ORANLARI', fontsize=25)
    plt.xlabel('İLÇELER', fontsize=20)
    plt.ylabel('YANGIN ORANI', fontsize=20)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=15)
    #plt.show()
    with st.expander("İlçe bazlı yangın tahmin oranları"):
        st.write("""
            İlçelerdeki gelecek yangın oranları, model tarafından üretilen değerler görselleştirilerek gösterilmektedir..
        """)
        st.pyplot(plt)



tempSumMonthGrid(tempSum)
tempSumStateGrid(tempSum)

with st.form("my_form"):
    col1, col2, col3 = st.columns(3)
    tempData = ""
    tempValue = ""
    tempLabel = ""
    with col1:
        tempStateCol = st.selectbox(
            'Hangi ilçe için tahmin verilerini inceleyeceksiniz',
            ('ARNAVUTKOY', 'ATASEHIR', 'BEYKOZ', 'BEYOGLU', 'CATALCA', 'ESENYURT', 'FATIH', 'KADIKOY', 'KAGITHANE', 'UMRANIYE', 'USKUDAR', 'ZEYTINBURNU'))

    with col2:
        tempMonthCol = st.selectbox(
            'Hangi ay için tahmin verilerini inceleyeceksiniz',
            ('OCAK', 'SUBAT', 'MART', 'NISAN', 'MAYIS', 'HAZIRAN', 'TEMMUZ', 'AGUSTOS', 'EYLUL', 'EKIM', 'KASIM', 'ARALIK'))



    submittedx = st.form_submit_button("Submit")
    if submittedx:
        for x in range(len(tempSum['tempSS'])):
            if(tempSum['xTestState'][x] == tempStateCol and tempSum['xTestMonth'][x] == tempMonthCol ):
                tempValue = tempSum['tempSS'][x]

        tempData = tempValue-tempMean
        tempLabel = tempStateCol + " " + tempMonthCol

        with col3:
            st.metric(label=tempLabel, value=tempValue, delta=tempData)

mainBase()
