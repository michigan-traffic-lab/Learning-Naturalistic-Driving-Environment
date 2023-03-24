"""
This file is to calculate crash rate.
"""
import os
import numpy as np
import pandas as pd


def analyze_collision_rate(res_folder, output_folder):
    res_df_list = []
    for subfolder in sorted(os.listdir(res_folder)):
        data_path = os.path.join(res_folder, subfolder, "res_df.csv")
        try:
            res_df_list.append(pd.read_csv(data_path))
        except:
            raise ValueError("No file: {0}".format(data_path))
    res_df = pd.concat(res_df_list)

    total_sim_num = res_df['Sim number'].sum()
    total_colli_num = res_df['colli num'].sum()
    total_travel_distances = res_df['total travel dist'].sum()  # unit: meter
    total_sim_wall_time = res_df['total sim wall time'].sum()  # unit: second

    cr_crash_per_km = total_colli_num / (total_travel_distances / 1000)

    sum_res_df = pd.DataFrame(np.array([total_sim_num, total_colli_num, cr_crash_per_km, total_sim_wall_time, total_travel_distances]).reshape(1, -1),
                              columns=['Sim number', 'colli num', 'cr (crash/km)', 'total sim wall time (s)', 'total travel dist (m)'])

    save_path = os.path.join(output_folder, 'collision-rate.csv')
    sum_res_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    # Simulation results
    res_folder = r'./raw_data/NeuralNDE/'
    output_folder = os.path.join('plot/crash_rate')
    os.makedirs(output_folder, exist_ok=True)

    # Analyze the results
    analyze_collision_rate(res_folder, output_folder)
