from dataLoader import get_raw_data

import keras
import keras.layers as layers

def createModel(inputSize, timesteps, hiddenSize):
    rnn = keras.models.Sequential()

    inputLayer = layers.SimpleRNN(hiddenSize,
        input_shape=(timesteps, inputSize),
        name='input')

    recurrentLayer = layers.SimpleRNN(
        inputLayer,
        return_sequences=True, 
        unroll=True, 
        name='rnn_layer')
    rnn.add(recurrentLayer)

    rnn.compile(optimizer='adam', loss='mse')

    return rnn


def trainModel(model, dataGenerator):
    model.fit_generator(dataGenerator)   

def evalModel(model):
    print('Score is 5')


def main():
    timesteps = 50

    model = createModel(4, timesteps, 5)

    data, targets = get_raw_data(predictionHoizon=1)

    dataGenerator = keras.preprocessing.sequence.TimeseriesGenerator(
        data, targets, timesteps, sampling_rate=1, stride=50, start_index=0, 
        end_index=None, shuffle=False, reverse=False, batch_size=128)

    trainModel(model, dataGenerator)

    evalModel(model)


main()