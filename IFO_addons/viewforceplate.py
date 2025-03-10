import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#read from txt in forceplatedata folder




def view_data(file_name):

    data=pd.read_table(file_name,delimiter=',',header=None,names=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
    print(data)
    sampling_freq = 1000  # Hz
    time = np.arange(len(data)) / sampling_freq
    data['Time'] = time
    print(data.head())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot forces
    ax1.plot(data['Time'], data['Fx'], label='Fx')
    ax1.plot(data['Time'], data['Fy'], label='Fy')
    ax1.plot(data['Time'], data['Fz'], label='Fz')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (N)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Force Components')

    # Plot moments
    ax2.plot(data['Time'], data['Mx'], label='Mx')
    ax2.plot(data['Time'], data['My'], label='My')
    ax2.plot(data['Time'], data['Mz'], label='Mz')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Moment (Nm)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Moment Components')

    plt.tight_layout()
    plt.show()

    return data

def sync_data(file_name,data,syncpoint,endpoint):

    data = data[data['Time'] >= syncpoint]
    #remove everything after endpoint
    data = data[data['Time'] <= endpoint]
    data['Time'] = data['Time'] - syncpoint

    #make time first column
    data = data[['Time', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']]
    #remove index
    data.to_csv('synced_data.csv', index=False)

    
    view_data('synced_data.txt')
    return data

def sync_trcs_and_mots(mot_file,video_sync_point,video_end_point):
    #read trc file
    data = pd.read_csv(mot_file, delimiter='\t', header=8)
    print(data.head())

    #remove everything from before sync point
   # trc_data = trc_data[trc_data['Time'] >= video_sync_point]
    #remove everything after endpoint
   # trc_data = trc_data[trc_data['Time'] <= video_end_point]

    data=data[data['time'] >= video_sync_point]
    data=data[data['time'] <= video_end_point]
    data['time']=data['time']-video_sync_point

    # for column in data.select_dtypes(include=['float64']).columns:
    #     data[column] = data[column].map('{:.8f}'.format)


    nRows = len(data)
    nColumns = len(data.columns)

    header_text = f"""Coordinates
version=1
nRows={nRows}
nColumns={nColumns}
inDegrees=yes

Units are S.I. units (second, meters, Newtons, ...)
If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).

endheader"""

    # Then save as above
    with open('output_file.mot', 'w') as f:
        f.write(header_text + '\n')
        data.to_csv(f, sep='\t', index=False, float_format='%.8f', line_terminator='\n')


#split data
def splitdata(synced_data):
    df=synced_data
    new_df = pd.DataFrame()
    new_df['time'] = df['Time']

    fx_median = df['Fx'].median()
    fy_median = df['Fy'].median()
    fz_median = df['Fz'].median()

    # Calculate deviations from median for each component
    fx_dev = df['Fx'] - fx_median
    fy_dev = df['Fy'] - fy_median
    fz_dev = df['Fz'] - fz_median

    # Calculate distribution factors for each component
    # For Fx (medial-lateral)
    right_factor_x = 0.5 + (fx_dev / (2 * np.abs(fx_dev).max()))
    left_factor_x = 1 - right_factor_x

    # For Fy (anterior-posterior)
    right_factor_y = 0.5 + (fy_dev / (2 * np.abs(fy_dev).max()))
    left_factor_y = 1 - right_factor_y

    # For Fz (vertical)
    right_factor_z = 0.5 + (fz_dev / (2 * np.abs(fz_dev).max()))
    left_factor_z = 1 - right_factor_z



    # Distribute forces
    # Right foot
    new_df['ground_force_r_vx'] = df['Fx'] * right_factor_x
    new_df['ground_force_r_vy'] = df['Fy'] * right_factor_y
    new_df['ground_force_r_vz'] = df['Fz'] * right_factor_z

    # Left foot
    new_df['ground_force_l_vx'] = df['Fx'] * left_factor_x
    new_df['ground_force_l_vy'] = df['Fy'] * left_factor_y
    new_df['ground_force_l_vz'] = df['Fz'] * left_factor_z

    # Calculate CoP and moments (assuming we have these columns in original data)
    # Right foot

    new_df['ground_torque_r_x'] = df['Mx'] * right_factor_x
    new_df['ground_torque_r_y'] = df['My'] * right_factor_y
    new_df['ground_torque_r_z'] = df['Mz'] * right_factor_z

    new_df['ground_torque_l_x'] = df['Mx'] * left_factor_x
    new_df['ground_torque_l_y'] = df['My'] * left_factor_y
    new_df['ground_torque_l_z'] = df['Mz'] * left_factor_z

    epsilon = 1e-10
    r_copx = -df['My'] * right_factor_y / (df['Fz'] * right_factor_z + epsilon)
    r_copy = df['Mx'] * right_factor_x / (df['Fz'] * right_factor_z + epsilon)
    r_copz = np.zeros_like(r_copx)  # Usually zero as COP is on force plate surface

    # Calculate COP for left foot
    l_copx = -df['My'] * left_factor_y / (df['Fz'] * left_factor_z + epsilon)
    l_copy = df['Mx'] * left_factor_x / (df['Fz'] * left_factor_z + epsilon)
    l_copz = np.zeros_like(l_copx)

    # Add COP to dataframe
    new_df['ground_force_r_px'] = r_copx
    new_df['ground_force_r_py'] = r_copy
    new_df['ground_force_r_pz'] = r_copz
    new_df['ground_force_l_px'] = l_copx
    new_df['ground_force_l_py'] = l_copy
    new_df['ground_force_l_pz'] = l_copz


    # Save to new file
    new_df = new_df[['time',
                    'ground_force_r_vx', 'ground_force_r_vy', 'ground_force_r_vz',
                    'ground_force_r_px', 'ground_force_r_py', 'ground_force_r_pz',
                    'ground_torque_r_x', 'ground_torque_r_y', 'ground_torque_r_z',
                    'ground_force_l_vx', 'ground_force_l_vy', 'ground_force_l_vz',
                    'ground_force_l_px', 'ground_force_l_py', 'ground_force_l_pz',
                    'ground_torque_l_x', 'ground_torque_l_y', 'ground_torque_l_z']]

    header_text = f"""name force_plate_data
    datacolumns {len(new_df.columns)}
    datarows {len(new_df)}
    version 1
    nRows {len(new_df)}
    nColumns {len(new_df.columns)}
    inDegrees no

    endheader
    """

    # Write both header and data in one operation
    with open('formatted_forces.mot', 'w') as f:
        f.write(header_text)
        new_df.to_csv(f, sep='\t', index=False, float_format='%.8f', line_terminator='\n')


#run main loop
if __name__ == "__main__":
    file_name = 'Forceplatedata/IFO_Trial_1_C.txt'
    video_file='C:/Python_directory/Repos/opencap/opencap-core/Data/Isabelle/OpenSimData/Kinematics/Trial_1_C.mot'
    syncpoint = 11.70  
    endpoint = 36.06  
    data=view_data(file_name)
    synced_data = sync_data(file_name,data, syncpoint, endpoint)
    video_sync_point = 5.831 
    video_end_point = 30.373 
    sync_trcs_and_mots(video_file,video_sync_point,video_end_point)
    splitdata(synced_data)


# fig, axes = plt.subplots(3, 1, figsize=(12, 12))
# fig.suptitle('Force Distribution Between Left and Right Feet')

# # Plot vertical forces (Fz)
# axes[0].plot(new_df['time'], new_df['ground_force_r_vz'], 'b-', label='Right Foot Fz')
# axes[0].plot(new_df['time'], new_df['ground_force_l_vz'], 'r-', label='Left Foot Fz')
# axes[0].plot(new_df['time'], df['Fz'], 'k--', label='Total Fz')
# axes[0].set_ylabel('Vertical Force (N)')
# axes[0].legend()
# axes[0].grid(True)

# # Plot anterior-posterior forces (Fy)
# axes[1].plot(new_df['time'], new_df['ground_force_r_vy'], 'b-', label='Right Foot Fy')
# axes[1].plot(new_df['time'], new_df['ground_force_l_vy'], 'r-', label='Left Foot Fy')
# axes[1].plot(new_df['time'], df['Fy'], 'k--', label='Total Fy')
# axes[1].set_ylabel('Anterior-Posterior Force (N)')
# axes[1].legend()
# axes[1].grid(True)

# # Plot medial-lateral forces (Fx)
# axes[2].plot(new_df['time'], new_df['ground_force_r_vx'], 'b-', label='Right Foot Fx')
# axes[2].plot(new_df['time'], new_df['ground_force_l_vx'], 'r-', label='Left Foot Fx')
# axes[2].plot(new_df['time'], df['Fx'], 'k--', label='Total Fx')
# axes[2].set_ylabel('Medial-Lateral Force (N)')
# axes[2].set_xlabel('Time (s)')
# axes[2].legend()
# axes[2].grid(True)

# plt.tight_layout()
# plt.show()