import pandas as pd

from createModel import *


dataset = pd.read_csv('amazon.csv', encoding='unicode_escape', sep=';')

create(dataset)

features = dataset[['state', 'month']]
targets = dataset['number']

for x in range(len(dataset['number'])):
    if(dataset['number'][x]<0):
        print("AAAAAAAA")


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
model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test),
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
print(temp)
