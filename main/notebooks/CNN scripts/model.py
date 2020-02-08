# Inspired from https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57

from keras.models import Model, Sequential
from keras.layers import Lambda, Dense, Dropout, Input, Concatenate
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import LSTM
from keras.backend import expand_dims, squeeze

# Base model
# 3 of them are run in parallel with each a downsampled version of the data
def get_base_model(input_len, kernel_size):
    nb_filters = 128

    model = Sequential([
#        Conv1D(nb_filters, kernel_size, activation='relu', input_shape=(input_len, 1)),
#        Conv1D(nb_filters, kernel_size, activation='relu'),
#        MaxPooling1D(4),
#        Conv1D(nb_filters*2, kernel_size, activation='relu'),
#        Conv1D(nb_filters*2, kernel_size, activation='relu'),
#        GlobalMaxPooling1D(),
#        MaxPooling1D(8),
#        Lambda(lambda y: squeeze(y, axis=0)),
        LSTM(600),
        Dense(50, activation='relu'),
        #Dropout(0.1),
    ])

    return model

# Main model
#def main_model(inputs_lens = [250, 625, 1250, 10], kernel_sizes = [4, 8, 16]):
def main_model(inputs_lens = [250, 625, 1250], kernel_sizes = [4, 8, 16]):
    # the inputs to the branches are the original time series, and its down-sampled versions
    input_smallseq = Input(shape=(inputs_lens[0], 1), name='eeg_small')
    input_medseq   = Input(shape=(inputs_lens[1], 1), name='eeg_medium')
    input_origseq  = Input(shape=(inputs_lens[2], 1), name='eeg_original')

    # auxiliary data
    #input_aux      = Input(shape=(inputs_lens[3],), name='aux')

    # the more down-sampled the time series, the shorter the corresponding filter
    base_net_small    = get_base_model(inputs_lens[0], kernel_sizes[0])
    base_net_med      = get_base_model(inputs_lens[1], kernel_sizes[1])
    base_net_original = get_base_model(inputs_lens[2], kernel_sizes[2])

    embedding_small    = base_net_small(input_smallseq)
    embedding_med      = base_net_med(input_medseq)
    embedding_original = base_net_original(input_origseq)

    # concatenate all the outputs
    x = Concatenate()([embedding_small, embedding_med, embedding_original])
    #x = Dropout(0.1)(x)

    # Conv layer
#    x = Lambda(lambda y: expand_dims(y, axis=-1))(x)
#    x = Conv1D(256, 4, activation='relu')(x)
#     x = GlobalMaxPooling1D()(x)
#     x = MaxPooling1D(10)(x)
#     x = Lambda(lambda y: squeeze(y, axis=0))(x)
#    x = LSTM(150)(x)
#    x = Dropout(0.3)(x)

    # auxiliary output
    #aux_out = Dense(3, activation='softmax')(x)

    # merge with aux data
    #merged_with_aux = Concatenate()([x, input_aux])

    # dense architecture
    #x = Dense(64, activation='relu')(x)
    #stacked = Dense(128, activation='relu')(stacked)
    #stacked = Dense(128, activation='relu')(stacked)

    # output layer
    #out = Dense(3, activation='softmax')(stacked)
    out = Dense(3, activation='softmax')(x)

 #   model = Model(
  #      inputs=[input_smallseq, input_medseq, input_origseq, input_aux],
   #     outputs=[aux_out, out])
#     model = Model(
 #        inputs=[input_smallseq, input_medseq, input_origseq, input_aux],
  #       outputs=out)
    model = Model(
        inputs=[input_smallseq, input_medseq, input_origseq],
        output=out)

    return model
