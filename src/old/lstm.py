from src.dataLoader.planets import get_raw_data

import keras
import keras.layers as layers

def createModel(inputSize, timesteps, hiddenSize):
    rnn = keras.models.Sequential()

    inputLayer = layers.LSTM(
        32,
        return_sequences=True,
        input_shape=(timesteps, inputSize))
    rnn.add(inputLayer)

    recurrentLayer = layers.LSTM(
        4)
    rnn.add(recurrentLayer)

    #rnn.add(keras.layers.Flatten())

    rnn.compile(optimizer='adam', loss='mse')

    return rnn


def trainModel(model, dataGenerator):
    model.fit_generator(dataGenerator)   
    #X, Y = dataGenerator[0]
    #model.fit(x=X,y=Y)

def evalModel(model):
    print('Score is 5')


def main():
    timesteps = 50

    model = createModel(4, timesteps, 5)

    data, targets = get_raw_data(predictionHoizon=1)

    dataGenerator = keras.preprocessing.sequence.TimeseriesGenerator(
        data, targets, timesteps, sampling_rate=1, stride=50, start_index=0, 
        end_index=None, shuffle=False, reverse=False, batch_size=128)

    for _ in range(100):
        trainModel(model, dataGenerator)

    evalModel(model)


main()