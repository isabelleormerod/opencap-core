import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt
from scipy.signal import find_peaks
import chardet

def process_text_file(file_name, sampling_frequency=1000):
    """
    Process a text file to convert it into a DataFrame and save it as CSV.

    """
    emg_signal = pd.read_csv(file_name + '.txt', sep="\t", skiprows=5, header=0,encoding='UTF-16-le').values
    emg_signal = pd.DataFrame(emg_signal, columns=['left', 'right'])
    emg_signal['Time'] = emg_signal.index / sampling_frequency

    #remove nan value rows
    emg_signal.dropna(inplace=True)
    emg_signal.to_csv(file_name + '.csv', index=False)
    return emg_signal

def notch_filter(data, fs=1000, f0=50, Q=30):
    """
    Apply a notch filter to remove powerline noise.
    """
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data)

def bandpass_filter(data, fs=1000, lowcut=20, highcut=450, order=4):
    """
    Apply a bandpass filter to the data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def rms_filter(data, frame=500):
    """
    Apply an RMS filter to smooth the data.
    """
    # Bandpass filter the data
    filtered_data = bandpass_filter(data)

    # Calculate RMS using a rolling window
    df = pd.DataFrame(filtered_data, columns=['Filtered'])
    df['RMS'] = df['Filtered'].rolling(window=frame, min_periods=1).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True
    )
    return df['RMS'].values

def calculate_global_min_max(file):
    """
    Calculate the global minimum and maximum across RMS_left and RMS_right
    in both small and big datasets.
    """
    # Read the small and big datasets
    data = pd.read_csv(file)

    
    # Combine RMS values from both datasets
    left = pd.concat([data['RMS_left'], data['RMS_left']])
    right = pd.concat([data['RMS_right'], data['RMS_right']])
    
    # Calculate global min and max
    global_min = min(left.min(), right.min())
    global_max = max(left.max(), right.max())
    
    return global_min, global_max


def normalise_with_global_range(data, global_min, global_max):
    """
    Normalize data using a global min and max range.
    """
    return (data - global_min) / (global_max - global_min)

def create_features(file_name,output_repo,EMG_repo):
    """
    Process a CSV file to extract features and save the output.
    """
    emg_data = pd.read_csv(file_name)

    # Apply filters and extract features
    emg_data['Bandpass_Filtered_left'] = bandpass_filter(emg_data['left'])
    emg_data['Bandpass_Filtered_right'] = bandpass_filter(emg_data['right'])
    emg_data['RMS_left'] = rms_filter(emg_data['left'])
    emg_data['RMS_right'] = rms_filter(emg_data['right'])


    # Save the processed data
    output_file = file_name.replace('.csv', '_filtered.csv')
    output_file=file_name.replace(EMG_repo, output_repo)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  
    print(output_file)
    emg_data.to_csv(output_file, index=False)

    return output_file


def normalise_rms_values(emg_file_name,file_name, output_repo, global_min, global_max):
        file_name = pd.read_csv(file_name)
        file_name['Normalised_RMS_left'] = normalise_with_global_range(file_name['RMS_left'], global_min, global_max)
        file_name['Normalised_RMS_right'] = normalise_with_global_range(file_name['RMS_right'], global_min, global_max)
        file_name.to_csv(f'{output_repo}/{emg_file_name}_filtered_normalised.csv', index=False)


# Main processing loop
Participant_numbers = [1,3,4,6,7,8,9]  # List of participant numbers to process
EMG_repo = 'EMG_data/'
output_repo='EMG_data/Filtered/'

base_dir = "E:/Rowing_Dataset/"  # Change this to your desired location
Trial_names=['High1','High2','Low1','Low2','Medium1','Medium2']


##FILTERING##
# for Participant_number in Participant_numbers:
#     for trial in Trial_names:
#         emg_file_name=f'{base_dir}Participant_{Participant_number}/P{Participant_number}_{trial}'
#         process_text_file(emg_file_name)
#         EMG_repo=f'{base_dir}Participant_{Participant_number}/'
#         output_repo=f'{base_dir}Participant_{Participant_number}/Filtered_EMG/'
#         emg_file=os.path.join(f'{emg_file_name}.csv')
#         emg_file_filtered=create_features(emg_file,
#                                           output_repo,EMG_repo)
        # global_min,global_max=calculate_global_min_max(emg_file_filtered)
        # normalise_rms_values(emg_file_name,emg_file_filtered,output_repo,global_min, global_max)


##PLOTTING##
for Participant_number in Participant_numbers:
    for trial in Trial_names:
        for arm in ['right','left']:
            EMG_repo=f'{base_dir}Participant_{Participant_number}/Filtered_EMG/'
            emg_file_filtered=os.path.join(f'{EMG_repo}P{Participant_number}_{trial}.csv')   
            df=pd.read_csv(emg_file_filtered)
            plt.plot(df['Time'],df[f'RMS_{arm}'])
            df['Trim start']=0
            df['Trim end']=60.0
          #  df=df[df['Time']>df['Trim start']]
          #  df=df[df['Time']<df['Trim end']]
            plt.plot(df['Time'],df[f'RMS_{arm}'])

            troughs, properties = find_peaks(-df[f'RMS_{arm}'],
                                            prominence=8,  # Adjust this threshold
                                            distance=800)     # Minimum samples between troughs

                # Plot
            plt.figure(figsize=(12,6))
            plt.plot(df['Time'], df[f'RMS_{arm}'], label=f'RMS_{arm}')
                # Plot the troughs
            plt.plot(df['Time'].iloc[troughs], df[f'RMS_{arm}'].iloc[troughs], "x", label='Troughs')
            trough_times = df['Time'].iloc[troughs]
            trough_values = df['RMS_left'].iloc[troughs]
            plt.show()

            #found peaks, discard first 5 and last one
            df=df[df['Time']<df['Time'].iloc[troughs[-1]]]
            df=df[df['Time']>df['Time'].iloc[troughs[3]]]

        #split up by peak
        troughs, properties = find_peaks(-df[f'RMS_{arm}'],
                                            prominence=8,  # Adjust this threshold
                                            distance=800)
        stroke_data = []

        # Loop through troughs to create segments
        for n in range(len(troughs)-1):
            start_idx = troughs[n]
            end_idx = troughs[n+1]

            # Create dictionary for this stroke
            stroke = {
                'Stroke': n+1,
                f'EMG_{arm}': df[f'RMS_{arm}'].iloc[start_idx:end_idx].values,
                'Time': df['Time'].iloc[start_idx:end_idx].values
            }

            stroke_data.append(stroke)

        # Convert to DataFrame
        trough_info = pd.DataFrame(stroke_data)
        print(trough_info)

        #plot stroke 1
        plt.plot(trough_info['Time'][0], trough_info[f'EMG_{arm}'][0])
        trough_info.to_csv(f'{EMG_repo}P{Participant_number}_{trial}_{arm}_per_peak.csv', index=False)
        plt.show()

        

        



