import pandas as pd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt

mot_repo='E:/Open_cap/'
session_IDs=['Gabriella Miles','Margo Bellamy','Rob Ballentyne','Aman Kukreja','Adam McClenaghan','Varun Ramanan','Harrison Mogg-Wells','Jonathan Erksine','Owen Peckham','Jonny Raines']
session_IDs=['Rob Ballentyne']

small_handle_trials=['Small_handle_trial_1','Small_handle_trial_2','Small_handle_trial_3','Small_handle_trial_4']
big_handle_trials=['Big_handle_trial_1','Big_handle_trial_2','Big_handle_trial_3','Big_handle_trial_4']
# session_IDs=['Adam McClenaghan']
# small_handle_trials=['Small_handle_trial_2']
# big_handle_trials=['Big_handle_trial_2']
results = pd.DataFrame()
trials = list(zip(small_handle_trials, big_handle_trials))  # Convert to list to allow multiple iterations


# def find_trial_IDs(session_IDs, trials, base_path):
#     """
#     Function to find trial IDs where .mov files are found in the small and big camera directories.

#     Parameters:
#         session_IDs (list): List of session IDs.
#         trials (list of tuples): List of trial name pairs (small_trial, big_trial).
#         base_path (str): Base file path to construct the directories.

#     Returns:
#         list: A 2D list where the first column contains full paths for small camera .mov files,
#               and the second column contains full paths for big camera .mov files.
#     """
#     trial_IDs = []

#     for session_ID in session_IDs:
#         for small_trial, big_trial in trials:
#             # Construct the paths to the small and big camera directories
#             small_camera_directory = os.path.join(
#                 base_path, f"{session_ID}/Videos/Cam0/InputMedia/{small_trial}/"
#             )

#             big_camera_directory = os.path.join(
#                 base_path, f"{session_ID}/Videos/Cam1/InputMedia/{big_trial}/"
#             )

#             small_file_path = None
#             big_file_path = None

#             # Check if the small camera directory exists and find the first .mov file
#             if os.path.exists(small_camera_directory):
#                 for small_ID in os.listdir(small_camera_directory):
#                     if small_ID.endswith('.mov'):
#                         small_path = os.path.join(small_camera_directory, small_ID)
#                         small_ID = small_ID.split('.')[0]
#                         break  # Stop after finding the first .mov file

#             # Check if the big camera directory exists and find the first .mov file
#             if os.path.exists(big_camera_directory):
#                 for big_ID in os.listdir(big_camera_directory):
#                     if big_ID.endswith('.mov'):
#                         big_path = os.path.join(big_camera_directory, big_ID)
#                         big_ID = big_ID.split('.')[0]
#                         break  # Stop after finding the first .mov file

#             # Append the small and big trial paths as a pair to the list
#             trial_IDs.append([small_ID, big_ID])

#     return trial_IDs

def apply_butter_filter(data, fs=60, cutoff=20, order=4):
    """
    Apply a bandpass filter to the data.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    if not (0 < normalized_cutoff < 1):
        raise ValueError(f"Cutoff frequency must be between 0 and Nyquist frequency. Received: {cutoff}")

    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, data)

def filter_signals(columns, file_trial_IDs):
    small_df = pd.read_csv(f'{file_trial_IDs[0]}.mot', sep='\t', skiprows=10, header=0)
    big_df = pd.read_csv(f'{file_trial_IDs[1]}.mot', sep='\t', skiprows=10, header=0)

    for column in columns:
        if column not in small_df.columns or column not in big_df.columns:
            raise ValueError(f"Column '{column}' not found in the data.")

        # Apply rolling mean
        small_df[column] = small_df[column].rolling(window=5, min_periods=1).mean()
        big_df[column] = big_df[column].rolling(window=5, min_periods=1).mean()

        # Fill NaN values before filtering
        small_df[column] = small_df[column].fillna(method='bfill').fillna(method='ffill')
        big_df[column] = big_df[column].fillna(method='bfill').fillna(method='ffill')

        # Apply Butterworth filter
        small_df[column] = apply_butter_filter(small_df[column].values, fs=60, cutoff=10, order=4)
        big_df[column] = apply_butter_filter(big_df[column].values, fs=60, cutoff=10, order=4)

    # Save filtered data
    small_df.to_csv(f'{file_trial_IDs[0]}_filtered.csv', index=False)
    big_df.to_csv(f'{file_trial_IDs[1]}_filtered.csv', index=False)


def plot_comparisons(columns, file_trial_IDs,crop_start=None,crop_end=None):

    small_data = pd.read_csv(f'{file_trial_IDs[0]}_filtered.csv')
    big_data = pd.read_csv(f'{file_trial_IDs[1]}_filtered.csv')

    if crop_start is not None and crop_end is not None:
        small_data = small_data[(small_data['time'] >= crop_start) & (small_data['time'] <= crop_end)]
        big_data = big_data[(big_data['time'] >= crop_start) & (big_data['time'] <= crop_end)]


    # Plot Deltoid RMS
    for column in columns:
        plt.figure(figsize=(7, 5))
        plt.plot(small_data['time'], small_data[column], label='Small Handle', linestyle='--')
        plt.plot(big_data['time'], big_data[column], label='Big Handle')
        plt.title(f'{column} Angle Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle in degrees')
        
        plt.legend(loc='upper right')

        # Plot Bicep RMS
        # plt.subplot(2, 1, 2)
        # plt.plot(small_data['Time'], small_data['RMS_Bicep'], label='Small Dataset', linestyle='--')
        # plt.plot(big_data['Time'], big_data['RMS_Bicep'], label='Big Dataset')
        # plt.title('Bicep Muscle Activation Comparison')
        # plt.xlabel('Time (s)')
        # plt.ylabel('RMS Amplitude')
        # plt.legend()
        plt.savefig(f'{file_trial_IDs[0]}_{column}_position.svg')

        plt.tight_layout()
    #plt.show()

def plot_velocities(columns, file_trial_IDs,crop_start=None,crop_end=None):

    small_data = pd.read_csv(f'{file_trial_IDs[0]}_filtered.csv')

    big_data = pd.read_csv(f'{file_trial_IDs[1]}_filtered.csv')

        #find derivative of the angle
    small_data['velocity']=np.gradient(small_data[columns[0]],small_data['time'])
    big_data['velocity']=np.gradient(big_data[columns[0]],big_data['time'])

    small_data['velocity'] = apply_butter_filter(small_data['velocity'].values, fs=60, cutoff=2, order=4)
    big_data['velocity'] = apply_butter_filter(big_data['velocity'].values, fs=60, cutoff=2, order=4)

    #find magnitude
    small_data['magnitude']=np.sqrt(small_data['velocity']**2)
    big_data['magnitude']=np.sqrt(big_data['velocity']**2)

        #find the acceleration
    small_data['acceleration']=np.gradient(small_data['velocity'],small_data['time'])
    big_data['acceleration']=np.gradient(big_data['velocity'],big_data['time'])

    std_dev_small=np.std(small_data['velocity'])
    std_dev_big=np.std(big_data['velocity'])

    mean_small=np.mean(small_data['velocity'])
    mean_big=np.mean(big_data['velocity'])

    cv_small=abs(std_dev_small)/abs(mean_small)
    cv_big=abs(std_dev_big)/abs(mean_big)


    if crop_start is not None and crop_end is not None:
        small_data = small_data[(small_data['time'] >= crop_start) & (small_data['time'] <= crop_end)]
        big_data = big_data[(big_data['time'] >= crop_start) & (big_data['time'] <= crop_end)]

    plt.figure(figsize=(7, 6))

        # Plot Deltoid RMS
    plt.plot(small_data['time'], small_data['acceleration'], label='Small Handle', linestyle='--')
    plt.plot(big_data['time'], big_data['acceleration'], label='Big Handle')
    plt.title('Acceleration Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Degrees/second$^2$')
    plt.legend(loc='upper right')

    plt.savefig(f'{file_trial_IDs[0]}_acceleration.svg')

    plt.tight_layout()
   # plt.show()

    return cv_small,cv_big

def plot_y_vel(columns, file_trial_IDs,crop_start=None,crop_end=None):

    small_data = pd.read_csv(f'{file_trial_IDs[0]}.trc', sep='\t', skiprows=4, header=0)
    big_data = pd.read_csv(f'{file_trial_IDs[1]}.trc', sep='\t', skiprows=4, header=0)

        #find derivative of the angle
    small_data['velocity_y']=np.gradient(small_data[columns[0]],small_data[small_data.columns[1]])
    big_data['velocity_y']=np.gradient(big_data[columns[0]],big_data[small_data.columns[1]])
    small_data['velocity_x']=np.gradient(small_data[columns[1]],small_data[small_data.columns[1]])
    big_data['velocity_x']=np.gradient(big_data[columns[1]],big_data[small_data.columns[1]])
    small_data['velocity_z']=np.gradient(small_data[columns[2]],small_data[small_data.columns[1]])
    big_data['velocity_z']=np.gradient(big_data[columns[2]],big_data[small_data.columns[1]])

    small_data['velocity'] = np.sqrt(small_data['velocity_x']**2+small_data['velocity_y']**2+small_data['velocity_z']**2)
    big_data['velocity'] = np.sqrt(big_data['velocity_x']**2+big_data['velocity_y']**2+big_data['velocity_z']**2)

    #apply rolling average
    small_data['velocity'] = small_data['velocity'].rolling(window=5, min_periods=1).mean()
    big_data['velocity'] = big_data['velocity'].rolling(window=5, min_periods=1).mean()

    small_data['velocity'] = apply_butter_filter(small_data['velocity'].values, fs=60, cutoff=2, order=4)
    big_data['velocity'] = apply_butter_filter(big_data['velocity'].values, fs=60, cutoff=2, order=4)

        #find the acceleration
    small_data['acceleration']=np.gradient(small_data['velocity'],small_data[small_data.columns[1]])
    big_data['acceleration']=np.gradient(big_data['velocity'],big_data[small_data.columns[1]])

    std_dev_small=np.std(small_data['velocity'])
    std_dev_big=np.std(big_data['velocity'])

    mean_small=np.mean(small_data['velocity'])
    mean_big=np.mean(big_data['velocity'])

    cv_small=abs(std_dev_small)/abs(mean_small)
    cv_big=abs(std_dev_big)/abs(mean_big)


    if crop_start is not None and crop_end is not None:
        small_data = small_data[(small_data[small_data.columns[1]] >= crop_start) & (small_data[small_data.columns[1]] <= crop_end)]
        big_data = big_data[(big_data[small_data.columns[1]] >= crop_start) & (big_data[small_data.columns[1]] <= crop_end)]

    plt.figure(figsize=(7, 5))

        # Plot Deltoid RMS
    plt.plot(small_data[small_data.columns[1]], small_data['acceleration'], label='Small Handle', linestyle='--')
    plt.plot(big_data[small_data.columns[1]], big_data['acceleration'], label='Big Handle')
    plt.title('Acceleration Comparison from Marker Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Degrees/second$^2$')
    plt.legend(loc='upper right')

    plt.savefig(f'{file_trial_IDs[0]}_acceleration.svg')

    plt.tight_layout()
   # plt.show()

    return cv_small,cv_big


for session_name in session_IDs:
    session_results = {'Session_ID': session_name} 
    for small,big in trials:

        print(f'Processing {session_name} - Small Trial: {small}, Big Trial: {big}')

 
        # Construct file paths
        file_name_small = f"{mot_repo}{session_name}/OpenSimData/Kinematics/{small}"
        file_name_big = f"{mot_repo}{session_name}/OpenSimData/Kinematics/{big}"
        trc_small=f"{mot_repo}{session_name}/MarkerData/PostAugmentation/{small}/{small}"
        trc_big=f"{mot_repo}{session_name}/MarkerData/PostAugmentation/{big}/{big}"

#'pelvis_rotation','pelvis_tilt','pelvis_list',

        columns = ['arm_add_r','lumbar_bending']

        filter_signals(columns, [file_name_small, file_name_big])

        plot_comparisons(columns, [file_name_small, file_name_big],10,59)

        # columns=['X59','Y59','Z59']
        # cv_small,cv_big=plot_y_vel(columns, [trc_small, trc_big],10,59)
        # print(f'Standard Deviation of Magnitude for Small Handle: {cv_small}')
        # print(f'Standard Deviation of Magnitude for Big Handle: {cv_big}')
        # print(f'Processed')

        #columns = ['elbow_flex_r']
        #plot_velocities(columns, [file_name_small, file_name_big],10,55)

        # session_results[f'{small}_StdDev'] = cv_small
        # session_results[f'{big}_StdDev'] = cv_big
    results = results.append(session_results, ignore_index=True)
    results.to_csv('standard_deviations.csv', index=False)