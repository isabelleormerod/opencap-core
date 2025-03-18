import numpy as np
import pandas as pd


trials=['Knee_Test_R','Knee_Test_L']

##looks like data is in terms of millivolts :( have contacted biometrics to try and sort this


def process_text_file(file_name, sampling_frequency=1000):
    """
    Process a text file to convert it into a DataFrame and save it as CSV.

    """
    signal = pd.read_csv(file_name + '.txt', sep="\t", skiprows=8, header=0,encoding='UTF-16-le').values
    signal = pd.DataFrame(signal, columns=['left', 'right','X','Y','DS'])
    signal['Time'] = signal.index / sampling_frequency

    magnitude = np.sqrt(signal['X']**2 + signal['Y']**2)
    signal['Magnitude'] = magnitude

    #remove nan value rows
    signal.dropna(inplace=True)
    signal.to_csv(file_name + '.csv', index=False)
    return signal

# def convert_to_degrees(signal):
#     """
#     Convert the signal to degrees.
#     """
#     signal['X'] = signal['X']
#     signal['Y'] = signal['Y']'
#     return signal


signal = process_text_file("E:/Rowing_Dataset/Left_Test/Knee_test_L")