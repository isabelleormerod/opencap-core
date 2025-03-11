import pandas as pd
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt


Repo='E:/Rowing_Dataset'
Participant_numbers=[1,2,3,4,6,7,8,9]
distance_interval=0.022


Low_trials=['Low1','Low2']
Medium_trials=['Med1','Med2']
High_trials=['High1','High2']
all_trials=Low_trials+Medium_trials+High_trials

for Participant_number in Participant_numbers:
    for trial_id in all_trials:

    #low trials
        Torque_data = pd.read_csv(f'{Repo}/Participant_{Participant_number}/Torque_Data/P{Participant_number}_{trial_id}.csv')
        num_strokes=Torque_data['stroke_number'].max()
        mega_df=pd.DataFrame()
        for i in range(num_strokes):
            df=pd.DataFrame()
            curve_info=Torque_data['curve_data'][Torque_data['stroke_number']==i+1]
            #print(curve_info)
            curve_list=curve_info.values.tolist()
            values = curve_info.iloc[0].split(',')
            values = [int(x) for x in values]
            df['Force']=values
            #average time and distance for now
            distance_data=[]
            time_data=[]
            stroke_data=[]
            participant=[]
            trial=[]
            accumilated_time=[]
            time_info=Torque_data['drive_time'][Torque_data['stroke_number']==i+1]
            recovery_info=Torque_data['recover_time'][Torque_data['stroke_number']==i+1]
            recovery_list=recovery_info.values.tolist()
            recovery_info=recovery_list[0]
            power_info=Torque_data['power'][Torque_data['stroke_number']==i+1]
            power_list=power_info.values.tolist()
            power_info=power_list[0]
            time_list=time_info.values.tolist()
            time_info=time_list[0]
            time_interval=time_info/len(df)
            df['Force difference']=df['Force'].diff()
            df['Calculated Time']=abs(df['Force difference']*distance_interval/power_info)
            df['Recovery Time']=recovery_info
            for j in range(len(df['Force'])):
                participant.append(Participant_number)
                trial.append(Low_trials[0])
                distance_data.append(distance_interval*j)
                time_data.append(time_interval*j)
                stroke_data.append(i+1)
            df['Force difference']=df['Force'].diff()
            df['Calculated Time']=abs(df['Force difference']*distance_interval/power_info)

            df['Velocity']=distance_interval/df['Calculated Time']
            df['Participant']=participant
            df['Trial']=trial
            df['Distance']=distance_data  
            df['Time_average']=time_data
            df['Stroke']=stroke_data

            #replace all nans with 0
            df.replace([np.inf, -np.inf], np.nan,inplace=True)
            df.fillna(0,inplace=True)

            total_time=df['Calculated Time'].sum()



        
            mega_df=mega_df.append(df)

        mega_df.to_csv(f'{Repo}/Participant_{Participant_number}/Torque_Data/P{Participant_number}_{trial_id}_with_curve.csv',index=False)

# plotting

# remove first 5 strokes in every trial
for participant in Participant_numbers:
    for trial_id in all_trials:
        Torque_data = pd.read_csv(f'{Repo}/Participant_{participant}/Torque_Data/P{participant}_{trial_id}_with_curve.csv')
        Torque_data=Torque_data[Torque_data['Stroke']>5]
        Torque_data.to_csv(f'{Repo}/Participant_{participant}/Torque_Data/P{participant}_{trial_id}_with_curve_cropped.csv',index=False)

participants=[1,2,3,4,6,7,8,9]
all_trials=['High1']
colors = plt.cm.tab20(np.linspace(0, 1, len(participants) * len(all_trials)))
color_index = 0
#plotting

#plot all force curves for participant 1
for participant in participants:
    for trial_id in all_trials:
        current_color = colors[color_index]
        Torque_data = pd.read_csv(f'{Repo}/Participant_{participant}/Torque_Data/P{participant}_{trial_id}_with_curve_cropped.csv')
        num_strokes=Torque_data['Stroke'].unique()
        for i in num_strokes:
            print(i)
            df=Torque_data[Torque_data['Stroke']==i+i]
        for i in num_strokes:
            df=Torque_data[Torque_data['Stroke']==i+i]
            if i == num_strokes[0]:
                plt.plot(df['Distance'], df['Force'],
                        color=current_color,
                        label=f'P{participant} - {trial_id}')
            else:
                plt.plot(df['Distance'], df['Force'],
                        color=current_color)

        color_index += 1

plt.xlabel('Distance (m)')
plt.ylabel('Force (N)')
plt.legend()
plt.title(f'Force curve for participant {participant} in trial {trial_id}')
plt.show()