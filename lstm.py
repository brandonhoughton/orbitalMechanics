from singleLayer import get_data

import keras
import keras.layers as layers

def createModel(inputSize, ):
    rnn = keras.models.Sequential()

    inputLayer = layers.Input(shape=(inputSize,), batch_shape=(None,inputSize), name='input')
    rnn.add(inputLayer)

    recurrentLayer_1 = layers.recurrent.rnn(inputLayer)
    rnn.add(recurrentLayer_1)

    recurrentLayer_2 = layers.recurrent.rnn(recurrentLayer_1)
    rnn.add(recurrentLayer_2)

    

def createModel():
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

def trainModel(model, ):
    model.compile(optimizer='adam', loss='mse')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2)   