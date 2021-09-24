import tensorflow as tf
import pandas as pd
import numpy as np
data=pd.read_csv('diabetes.csv')
dataset=data.pop('Outcome')

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,dataset,test_size=0.3)

callback=myCallback()
model=tf.keras.Sequential()
#model.add(tf.keras.layers.Flatten()
model.add(tf.keras.layers.Dense(units=64,activation='relu'))
model.add(tf.keras.layers.Dense(units=32,activation='relu'))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,callbacks=[callback])

model.save('Diabetes')