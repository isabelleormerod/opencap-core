import pandas as pd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt
import opensim as osim
from scipy.signal import find_peaks

all_ids=['12dc67fd-1cde-4dec-81e6-7f4e579829b9',
         '1f7ec53e-aeec-4304-8411-2528754b768e','f41fe453-e784-4d85-b6bd-0d17d5caa635',
         '095bc85e-09df-4767-87c7-e425f6e0f0e9','cf606a7f-090a-4248-90a4-f847d1b12cce',
         'f209a117-43b8-46a4-8376-2adf4f022559','058d2311-c5a3-4196-bd5a-939535b7d5bb']

mot_repo='E:/Rowing_Dataset'
Participant_number=[1,3,4,6,7,8,9]

ids=[]

for i in range(len(Participant_number)):
    ids.append(all_ids[i])
print(ids)




Low_trials=['Low1','Low2']
Medium_trials=['Medium1','Medium2']
High_trials=['High1','High2']

results = pd.DataFrame()
trials = Low_trials+Medium_trials+High_trials  # Convert to list to allow multiple iterations
#trials=['High1']

def rotate_trc_data(input_file, output_file, angle):
    """
    Rotate TRC file data around specified axis
    """
    # Create rotation transform
    angle_rad = np.radians(angle)
    axis = osim.Vec3(0, 0, 1)  # Z-axis rotation
    rotation = osim.Rotation(angle, osim.Vec3(0, 0, 1))  # For Z-axis rotation
    transform = osim.Transform(rotation, osim.Vec3(0, 0, 0))

    # Read TRC file
    adapter = osim.TRCFileAdapter()
    table = adapter.read(input_file)
    markers_table = table['markers']
   # print(dir(markers_table))

    # Get dimensions
    nRows = markers_table.getNumRows()
    nCols = markers_table.getNumColumns()
    #print(markers_table.getColumnLabels())

   # print(f"Processing {nRows} frames of data")
   # print(f"Number of markers: {nCols}")
   # print(markers_table)

    df = pd.read_csv(input_file, sep='\t', skiprows=4)

    num_frames = markers_table.getNumRows()
    transform_df = pd.DataFrame(index=range(num_frames+1),columns=df.columns)
  #  print(df['Time'].astype(float))
   # print(transform_df)

    labels = markers_table.getColumnLabels()
    num_markers = len(labels)
    num_frames = len(df)

    for frame in range(num_frames):
        for i in range(num_markers):

            # Get original point
            point = osim.Vec3(
                float(df[f'X{i+1}'].iloc[frame]),
                float(df[f'Y{i+1}'].iloc[frame]),
                float(df[f'Z{i+1}'].iloc[frame])
            )

            # Rotate point
            rotated_point = transform.xformFrameVecToBase(point)

            rotated_numpy = rotated_point.to_numpy()
            data_row = frame + 1  # +2 because row 0 is headers, and frame starts from 0


            # Store the rotated coordinates in the correct row
           # print(rotated_numpy)
           # print(x_idx, y_idx, z_idx)
            transform_df[f'X{i+1}'].iloc[data_row] = rotated_numpy[0]  # X component
            transform_df[f'Y{i+1}'].iloc[data_row] = rotated_numpy[1]  # Y component
            transform_df[f'Z{i+1}'].iloc[data_row]= rotated_numpy[2]  # Z component
          #  print(x_idx+2, y_idx+2, z_idx+2)
        



  #  Read the header from the original file
    with open(input_file, 'r') as f:
        header_lines = [next(f) for _ in range(4)]

    # Save the complete DataFrame to .trc
    transform_df.to_csv(output_file, sep='\t', index=False, header=False)
    transform_df.to_csv(output_file.replace('.trc', '.csv'), index=False)


    return transform_df

def compare_athletes(participants, ids, trials, columns, crop_start=None, crop_end=None):
    print('Starting')
    files=[f"{mot_repo}/Participant_{participants[0]}/OpenCapData_{ids[0]}/OpenSimData/Kinematics/P{participants[0]}_{trials[0]}.mot",
    f"{mot_repo}/Participant_{participants[1]}/OpenCapData_{ids[1]}/OpenSimData/Kinematics/P{participants[1]}_{trials[0]}.mot"]

    data_1 = pd.read_csv(files[0], sep='\t', skiprows=10)
    data_2 = pd.read_csv(files[1], sep='\t', skiprows=10)

    print(data_1.columns)

    for file in files:
    
        if crop_start is not None and crop_end is not None:
            data_1 = data_1[(data_1['time'] >= crop_start) & (data_1['time'] <= crop_end)]
            data_2 = data_2[(data_2['time'] >= crop_start) & (data_2['time'] <= crop_end)]


        # Plot Deltoid RMS
        for column in columns:
            plt.figure(figsize=(7, 5))
            plt.plot(data_1['time'], data_1[column], label=f'Participant {participants[0]}', linestyle='--')
            plt.plot(data_2['time'], data_2[column], label=f'Participant {participants[1]}')
            plt.title(f'{column} Angle Comparison')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle in degrees')
            plt.legend(loc='upper right')

            # Plot Bicep RMS
            # plt.subplot(2, 1, 2)
            # plt.plot(data_1['Time'], data_1['RMS_Bicep'], label='Small Dataset', linestyle='--')
            # plt.plot(data_2['Time'], data_2['RMS_Bicep'], label='Big Dataset')
            # plt.title('Bicep Muscle Activation Comparison')
            # plt.xlabel('Time (s)')
            # plt.ylabel('RMS Amplitude')
            # plt.legend()
            plt.savefig(f'E:\Rowing_Dataset\Graph_assets\Participants{participants[0]}_and_{participants[1]}_{column}_comparison.svg')
            plt.tight_layout()
    plt.show()
            




       # print(f'Processing Participant {participant} - Trial: {trial}')


columns = ['elbow_flex_r']
#compare_athletes([4,9], ids, trials, columns)

#find start of videos

#find per peak

def data_per_stroke(mot_repo, participants, trials, ids, columns=['elbow_flex_r']):
    
    for i,participant in enumerate(participants):
        for trial in trials:
        
            print(ids[i])

            df=pd.read_csv(f"{mot_repo}/Participant_{participant}/OpenCapData_{ids[i]}/OpenSimData/Kinematics/P{participant}_{trial}.mot", sep='\t', skiprows=10)
            
            plt.plot(df['time'],df[f'elbow_flex_l'])
            troughs, properties = find_peaks(-df[f'elbow_flex_l'],
                                                    prominence=10,  # Adjust this threshold
                                                    distance=40)     # Minimum samples between troughs
            


            plt.figure(figsize=(12,6))
            plt.plot(df['time'], df[f'elbow_flex_l'], label=f'Elbow_Flex')

            plt.plot(df['time'].iloc[troughs], df[f'elbow_flex_l'].iloc[troughs], "x", label='Troughs')
          #  plt.show()
            
            stroke_data = []

            print(len(troughs))

            df=df[df['time']<df['time'].iloc[troughs[-1]]]
            df=df[df['time']>df['time'].iloc[troughs[3]]]

            troughs, properties = find_peaks(-df[f'elbow_flex_l'],
                                                    prominence=10,  # Adjust this threshold
                                                    distance=40)  
            

                # Loop through troughs to create segments
            for n in range(len(troughs)-1):
                start_idx = troughs[n]
                end_idx = troughs[n+1]

                    # Create dictionary for this stroke
                stroke = {
                        'Stroke': n+1,
                        f'elbow_flex_l': df[f'elbow_flex_l'].iloc[start_idx:end_idx].values,
                        'time': df['time'].iloc[start_idx:end_idx].values
                        }

                stroke_data.append(stroke)

                

                # Convert to DataFrame
                trough_info = pd.DataFrame(stroke_data)
               # print(trough_info)

                trough_info.to_csv(f'{mot_repo}/P{Participant_number}_{trials}_MOT_per_peak.csv', index=False)

    return trough_info

data_per_stroke(mot_repo,[1,3,4,6,7,8,9], trials, ids)