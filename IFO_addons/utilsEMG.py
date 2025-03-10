import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt
import chardet

def process_text_file(file_name, sampling_frequency=1000):
    """
    Process a text file to convert it into a DataFrame and save it as CSV.

    """
    emg_signal = pd.read_csv(file_name + '.txt', sep="\t", skiprows=5, header=0,encoding='UTF-16-le').values
    emg_signal = pd.DataFrame(emg_signal, columns=['Deltoid', 'Bicep'])
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

def calculate_global_min_max(file_small, file_big):
    """
    Calculate the global minimum and maximum across RMS_Deltoid and RMS_Bicep
    in both small and big datasets.
    """
    # Read the small and big datasets
    small_data = pd.read_csv(file_small)
    big_data = pd.read_csv(file_big)
    
    # Combine RMS values from both datasets
    combined_deltoid = pd.concat([small_data['RMS_Deltoid'], big_data['RMS_Deltoid']])
    combined_bicep = pd.concat([small_data['RMS_Bicep'], big_data['RMS_Bicep']])
    
    # Calculate global min and max
    global_min = min(combined_deltoid.min(), combined_bicep.min())
    global_max = max(combined_deltoid.max(), combined_bicep.max())
    
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
    emg_data['Bandpass_Filtered_Deltoid'] = bandpass_filter(emg_data['Deltoid'])
    emg_data['Bandpass_Filtered_Bicep'] = bandpass_filter(emg_data['Bicep'])
    emg_data['RMS_Deltoid'] = rms_filter(emg_data['Deltoid'])
    emg_data['RMS_Bicep'] = rms_filter(emg_data['Bicep'])


    # Save the processed data
    output_file = file_name.replace('.csv', '_filtered.csv')
    output_file=file_name.replace(EMG_repo, output_repo)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  
    print(output_file)
    emg_data.to_csv(output_file, index=False)

    return output_file


def normalise_rms_values(file_name, global_min, global_max):
        file_name = pd.read_csv(file_name)
        file_name['Normalised_RMS_Deltoid'] = normalise_with_global_range(file_name['RMS_Deltoid'], global_min, global_max)
        file_name['Normalised_RMS_Bicep'] = normalise_with_global_range(file_name['RMS_Bicep'], global_min, global_max)
        file_name.to_csv(emg_file_small_filtered, index=False)


def plot_combined_rms(file_small, file_big, crop_start=None, crop_end=None,labels=['Small Handle','Big Handle']):
    """
    Plot RMS of Deltoid and Bicep for small and big datasets on separate graphs,
    with optional cropping of the data.
    """
    small_data = pd.read_csv(file_small)
    big_data = pd.read_csv(file_big)



    # Apply cropping if specified
    if crop_start is not None and crop_end is not None:
        small_data = small_data[(small_data['Time'] >= crop_start) & (small_data['Time'] <= crop_end)]
        big_data = big_data[(big_data['Time'] >= crop_start) & (big_data['Time'] <= crop_end)]

    plt.figure(figsize=(8, 5))

    # Plot Deltoid RMS
    plt.subplot(2, 1, 1)
    plt.plot(small_data['Time'], small_data['RMS_Deltoid'], label=labels[0], linestyle='--')
    plt.plot(big_data['Time'], big_data['RMS_Deltoid'], label=labels[1])
    plt.title('Shoulder Muscle Activation Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Amplitude')
    plt.legend(loc='upper right')


    # Plot Bicep RMS
    plt.subplot(2, 1, 2)
    plt.plot(small_data['Time'], small_data['RMS_Bicep'], label=labels[0], linestyle='--')
    plt.plot(big_data['Time'], big_data['RMS_Bicep'], label=labels[1])
    plt.title('Bicep Muscle Activation Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Amplitude')


    plt.legend(loc='upper right')

    plt.tight_layout()

    #save picture in folder
    plt.savefig(file_small.replace('.csv', f'_combined_rms.svg'))
    plt.show()


# Main processing loop
Participant_numbers = [4]  # List of participant numbers to process
EMG_repo = 'EMG_data/'
output_repo='EMG_data/Filtered/'



for Participant_number in Participant_numbers:
    # Paths for small and big EMG files

    # for trial in list(range(1,6)):
    #     process_text_file(os.path.join(EMG_repo, f'Participant_{Participant_number}_Trial_{trial}_Small'))
    #     process_text_file(os.path.join(EMG_repo, f'Participant_{Participant_number}_Trial_{trial}_Big'))

    #     emg_file_small = os.path.join(EMG_repo, f'Participant_{Participant_number}_Trial_{trial}_Small.csv')
    #     emg_file_big = os.path.join(EMG_repo, f'Participant_{Participant_number}_Trial_{trial}_Big.csv')
        
        
    #     emg_file_small_filtered=create_features(emg_file_small,output_repo,EMG_repo)
    #     emg_file_big_filtered=create_features(emg_file_big,output_repo,EMG_repo)
    #     # global_min,global_max=calculate_global_min_max(emg_file_small_filtered, emg_file_big_filtered)
    #     # normalise_rms_values(emg_file_small_filtered, global_min, global_max)
    #     # normalise_rms_values(emg_file_big_filtered, global_min, global_max)
    #     # Normalise RMS values


    emg_file_small_filtered = f'EMG_data/Filtered/Participant_{Participant_number}_Trial_1_Small.csv'
    emg_file_big_filtered = f'EMG_data/Filtered/Participant_{Participant_number}_Trial_1_Big.csv'

    emg_file_small_last=f'EMG_data/Filtered/Participant_{Participant_number}_Trial_5_Small.csv'
    emg_file_big_last=f'EMG_data/Filtered/Participant_{Participant_number}_Trial_5_Big.csv'


    #comparisons to make: trial 1 vs trial 5, for small and for big
    #also small vs big (just pick one trial for each)

    # Plot combined RMS for small and big datasets with cropping example
    plot_combined_rms(
        emg_file_small_last,
        emg_file_big_last,crop_start=15,crop_end=60  # Example crop start time in seconds    # Example crop end time in seconds
    )

    # plot_combined_rms(
    #     emg_file_small_filtered,
    #     emg_file_small_last,crop_start=15,crop_end=60,labels=['Trial 1','Trial 5']  # Example crop start time in seconds    # Example crop end time in seconds
    # )

    # plot_combined_rms(
    #     emg_file_big_filtered,
    #     emg_file_big_last,crop_start=15,crop_end=60,labels=['Trial 1','Trial 5']  # Example crop start time in seconds    # Example crop end time in seconds
    # )

    


