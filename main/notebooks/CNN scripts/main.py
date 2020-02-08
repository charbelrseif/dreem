import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import CSVLogger
from model import main_model

# Split data
df_path = os.path.join(os.path.dirname(__file__), '..', 'df_train_CNN.h5')
df_train = pd.read_hdf(df_path)

var_to_pred = 'SO'

X_train, X_val, y_train, y_val = train_test_split(
    df_train.loc[:, df_train.columns != var_to_pred],
    df_train[var_to_pred],
    test_size=0.10,
    random_state=0,
    stratify=df_train[var_to_pred])

y_train = pd.DataFrame(y_train)
y_val = pd.DataFrame(y_val)

# convert labels to categorical one-hot encoding
y_train = to_categorical(y_train.values, num_classes=3)
y_val = to_categorical(y_val.values, num_classes=3)

# Build inputs
eeg_indexes = np.arange(10, df_train.shape[1] - 3)

X_train_eeg = X_train.iloc[:, eeg_indexes]
X_val_eeg = X_val.iloc[:, eeg_indexes]

# we feed 2 downsampled datasets + the original dataset = 3
model_train = [
    X_train_eeg.iloc[:, np.arange(0, 1250, 5)].values[:,:,None], # small
    X_train_eeg.iloc[:, np.arange(0, 1250, 2)].values[:,:,None], # medium
    X_train_eeg.values[:,:,None],                                # original
#    X_train.iloc[:, 0:10],                                       # auxiliary
]

model_val = [
    X_val_eeg.iloc[:, np.arange(0, 1250, 5)].values[:,:,None], # small
    X_val_eeg.iloc[:, np.arange(0, 1250, 2)].values[:,:,None], # medium
    X_val_eeg.values[:,:,None],                                # original
 #   X_val.iloc[:, 0:10],                                       # auxiliary
]

# Run model
batch_size = 256
epochs = 50

model = main_model()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

csv_logger = CSVLogger('log_%d_%d.csv' % (batch_size, epochs), append=True, separator=';')

#model.fit(model_train, [y_train, y_train],
 #         batch_size=batch_size,
  #        epochs=epochs,
   #       validation_data=(model_val, [y_val, y_val]))

model.fit(model_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(model_val, y_val),
          callbacks=[csv_logger])
