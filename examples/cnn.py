#------------------------------------    CNN   -----------------------------------

#TAMAMEN NUMERİC VERİLER


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split

# Örnek sayısal veri oluşturma
import numpy as np
X = np.random.rand(1000, 10, 1)  # 1000 örnek, her biri 10 sayısal özellikten oluşuyor
y = np.random.randint(2, size=(1000, 1))  # 1000 binary target

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")



#TAMAMEN KOTEGORİK VERİ OLURSA



from sklearn.preprocessing import OneHotEncoder

# Örnek kategorik veri oluşturma
categories = ['cat', 'dog', 'mouse', 'bird']
X = np.random.choice(categories, size=(1000, 10))  # 1000 örnek, her biri 10 kategorik özellikten oluşuyor
y = np.random.randint(2, size=(1000, 1))  # 1000 binary target

# Kategorik verileri one-hot encode etme
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], -1, len(categories))

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(10, len(categories))))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")



#YARI SAYISAL YARI KATEGORİK VERİLER

# Örnek sayısal ve kategorik veri oluşturma
numeric_data = np.random.rand(1000, 5)  # 1000 örnek, her biri 5 sayısal özellikten oluşuyor
categories = ['cat', 'dog', 'mouse', 'bird']
categorical_data = np.random.choice(categories, size=(1000, 5))  # 1000 örnek, her biri 5 kategorik özellikten oluşuyor
y = np.random.randint(2, size=(1000, 1))  # 1000 binary target

# Kategorik verileri one-hot encode etme
encoder = OneHotEncoder(sparse=False)
categorical_encoded = encoder.fit_transform(categorical_data)

# Sayısal ve kategorik verileri birleştirme
X = np.concatenate((numeric_data, categorical_encoded), axis=1).reshape(1000, 10, 1)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")


#-------------------------------------------  NN ------------------------------------------------------------


#TAMAMEN NUMERİC VERİLER


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

def create_model(input_shape, num_layers):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    for _ in range(num_layers - 1):
        model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Örnek sayısal veri oluşturma
X = np.random.rand(1000, 10)  # 1000 örnek, her biri 10 sayısal özellikten oluşuyor
y = np.random.randint(2, size=(1000, 1))  # 1000 binary target

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = create_model((10,), num_layers=3)

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")


#TAMAMEN KOTEGORİK VERİ OLURSA


from sklearn.preprocessing import OneHotEncoder

def create_model(input_shape, num_layers):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    for _ in range(num_layers - 1):
        model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Örnek kategorik veri oluşturma
categories = ['cat', 'dog', 'mouse', 'bird']
X = np.random.choice(categories, size=(1000, 10))  # 1000 örnek, her biri 10 kategorik özellikten oluşuyor
y = np.random.randint(2, size=(1000, 1))  # 1000 binary target

# Kategorik verileri one-hot encode etme
encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = create_model((X_encoded.shape[1],), num_layers=3)

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")




#YARI NUMERİC YARI KATEGORİC



def create_model(input_shape, num_layers):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    for _ in range(num_layers - 1):
        model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Örnek sayısal ve kategorik veri oluşturma
numeric_data = np.random.rand(1000, 5)  # 1000 örnek, her biri 5 sayısal özellikten oluşuyor
categories = ['cat', 'dog', 'mouse', 'bird']
categorical_data = np.random.choice(categories, size=(1000, 5))  # 1000 örnek, her biri 5 kategorik özellikten oluşuyor
y = np.random.randint(2, size=(1000, 1))  # 1000 binary target

# Kategorik verileri one-hot encode etme
encoder = OneHotEncoder(sparse=False)
categorical_encoded = encoder.fit_transform(categorical_data)

# Sayısal ve kategorik verileri birleştirme
X = np.concatenate((numeric_data, categorical_encoded), axis=1)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = create_model((X.shape[1],), num_layers=3)

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")







