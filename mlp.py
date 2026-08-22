import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

df = pd.read_csv('chords.csv')

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


#encoding
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
y_oh = to_categorical(y_enc)

#traintest
X_train, X_val, y_train, y_val = train_test_split(X, y_oh, test_size=0.2, random_state=42, stratify=y_enc)

#model building
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(84,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_chord_model.keras', monitor='val_loss', save_best_only=True)


#model training
print("training model....")
history = model.fit(X_train, y_train, epochs=200, validation_data = (X_val, y_val), batch_size=32, callbacks=[early_stop, checkpoint])


#tf file export
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_model = converter.convert()

with open('Chords_classifier.tflite', 'wb') as f:
    f.write(tf_model)

print("mlp ran successfully !!!")