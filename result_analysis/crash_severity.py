"""
This file is to plot crash severity results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import copy
from itertools import combinations
import math
from scipy.stats import entropy
from pathlib import Path

from crash_type import collision_check, collision_type_classification


def perfect_inelastic_crash_change_of_velocity(v1_obj, v2_obj):
    """
    Assume two vehicles have a perfect inelastic collision to calculate the velocity (conservation of momentum) after crash.
    Then, calculate the change of velocity (vector difference between original and after crash velocity) to evaluate the crash severity.
    """

    def cal_change_of_velocity(v_vec_before, v_vec_after):
        deltaV = np.linalg.norm(v_vec_before - v_vec_after)

        return deltaV

    v1_velocity, v1_heading, v1_mass = v1_obj.speed, v1_obj.speed_heading, v1_obj.mass
    v2_velocity, v2_heading, v2_mass = v2_obj.speed, v2_obj.speed_heading, v2_obj.mass

    v1_velocity_vec = np.array([v1_velocity * np.cos(math.radians(v1_heading)), v1_velocity * np.sin(math.radians(v1_heading))])
    v2_velocity_vec = np.array([v2_velocity * np.cos(math.radians(v2_heading)), v2_velocity * np.sin(math.radians(v2_heading))])
    original_momentum = v1_mass * v1_velocity_vec + v2_mass * v2_velocity_vec

    velocity_vec_after_crash = original_momentum / (v1_mass + v2_mass)
    velocity_heading_after_crash = np.arctan2(velocity_vec_after_crash[1], velocity_vec_after_crash[0])

    v1_deltaV = cal_change_of_velocity(v_vec_before=v1_velocity_vec, v_vec_after=velocity_vec_after_crash)
    v2_deltaV = cal_change_of_velocity(v_vec_before=v2_velocity_vec, v_vec_after=velocity_vec_after_crash)

    deltaV = max(v1_deltaV, v2_deltaV)

    return deltaV


def _cal_severity_and_type(collision_frame_path, onestep_b4_collision_frame_path, sim_resol):
    def _construct_v_obj(crash_frame_vlist, b4_crash_frame_vlist, sim_resol):

        def _cal_v_speed(v_obj_before, v_obj_later, sim_resol):
            travel_dist = np.linalg.norm(np.array([v_obj_before.location.x, v_obj_before.location.y]) - np.array([v_obj_later.location.x, v_obj_later.location.y]))
            speed = travel_dist / sim_resol
            return speed

        crash_vid_list = []
        for vehicle_pair in combinations(crash_frame_vlist, r=2):
            v1, v2 = vehicle_pair[0], vehicle_pair[1]
            v1_poly, v2_poly = v1.poly_box, v2.poly_box

            if v1_poly.intersects(v2_poly):
                collision_flag = True
                if v1.id not in crash_vid_list:
                    crash_vid_list.append(v1.id)
                if v2.id not in crash_vid_list:
                    crash_vid_list.append(v2.id)

        v_obj_list = []
        for vid in crash_vid_list:
            v_obj_before, v_obj_later = None, None
            for v in b4_crash_frame_vlist:
                if v.id == vid:
                    v_obj_before = v
            for v in crash_frame_vlist:
                if v.id == vid:
                    v_obj_later = v

            try:
                speed = _cal_v_speed(v_obj_before, v_obj_later, sim_resol=sim_resol)
            except:
                speed = 0.
            v_obj = copy.deepcopy(v_obj_later)
            v_obj.speed = speed
            v_obj.mass = 1  # mass of the vehicle. To calculate the crash severity using perfect inelastic crash (conservation of momentum)

            v_obj_list.append(v_obj)

        return v_obj_list

    def _map_deltaV_to_severity(deltaV, collision_type):
        """
        Use different thresholds to categorize deltaV to crash severity.
        Here the used thresholds are from: Richards, D. C. "Relationship between speed and risk of fatal injury: pedestrians and car occupants." (2010).
        """
        if collision_type == "rear_end" or collision_type == "head_on":  # "frontal_impact"
            slight = 11.  # mph
            serious = 23.
            fatal = 34.
        else:  # "side_impact"
            slight = 8.  # mph
            serious = 14.
            fatal = 24.

        severity = None
        if deltaV <= slight:
            severity = "No Injury"
        elif slight < deltaV <= serious:
            severity = "Minor Injury"
        elif serious < deltaV <= fatal:
            severity = "Serious Injury"
        elif deltaV >= fatal:
            severity = "Fatal Injury"

        return severity

    crash_frame_vlist = pickle.load(open(collision_frame_path, "rb"))
    b4_crash_frame_vlist = pickle.load(open(onestep_b4_collision_frame_path, "rb"))

    collision_flag, collision_vehicle_list = collision_check(vehicle_list=crash_frame_vlist)
    collision_type, _, _ = collision_type_classification(collision_vehicle_list)

    v_obj_list = _construct_v_obj(crash_frame_vlist, b4_crash_frame_vlist, sim_resol=sim_resol)

    deltaV = perfect_inelastic_crash_change_of_velocity(v1_obj=v_obj_list[0], v2_obj=v_obj_list[1])
    deltaV = deltaV * 2.23694  # m/s to mph
    severity = _map_deltaV_to_severity(deltaV, collision_type)

    return deltaV, severity, collision_type


def analyze_crash_severity(res_folder, output_folder):

    res_folder = Path(res_folder)
    traj_dirs = list(res_folder.glob('*/*/*.pickle'))
    traj_dirs = sorted(traj_dirs)

    traj_dirs_df = pd.DataFrame({'path': traj_dirs})
    traj_dirs_df['folder'] = traj_dirs_df['path'].apply(lambda x: x.parts[-3])
    traj_dirs_df['episode_frame'] = traj_dirs_df["path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    traj_dirs_df["episode_id"] = (traj_dirs_df['episode_frame'].apply(lambda x: x.split('-')[0])).astype(int)
    traj_dirs_df["frame_id"] = (traj_dirs_df['episode_frame'].apply(lambda x: x.split('-')[1])).astype(int)

    deltaV_list_all = []
    deltaV_front_impact, deltaV_side_impact = [], []
    deltaV_severity = {"No Injury": 0, "Minor Injury": 0, "Serious Injury": 0, "Fatal Injury": 0}
    for folder_id in traj_dirs_df.folder.unique():
        one_folder_data = traj_dirs_df[traj_dirs_df['folder'] == folder_id]
        for episode_id in one_folder_data.episode_id.unique():
            one_episode = one_folder_data[one_folder_data['episode_id'] == episode_id]
            collision_frame_path = one_episode[one_episode.frame_id == one_episode.frame_id.max()]['path'].values[0]  # The crash frame.
            onestep_b4_collision_frame_path = one_episode[one_episode.frame_id == (one_episode.frame_id.max() - 1)]['path'].values[0]  # One step before the crash frame.

            deltaV, severity, collision_type = _cal_severity_and_type(collision_frame_path, onestep_b4_collision_frame_path, sim_resol=0.4)

            deltaV_list_all.append(deltaV)
            deltaV_severity[severity] += 1
            if collision_type == "rear_end" or collision_type == "head_on":
                deltaV_front_impact.append(deltaV)
            else:
                deltaV_side_impact.append(deltaV)

    _plot_severity_fig(sim_severity_dict=deltaV_severity, output_folder=output_folder)


def _plot_severity_fig(sim_severity_dict, output_folder):
    severity_type = ["No Injury", "Minor Injury", "Serious Injury", "Fatal Injury"]

    gt_severity_dict = {"No Injury": 498, "Minor Injury": 22, "Serious Injury": 0, "Fatal Injury": 0}  # 520 crashes from 2016-2020, data from MTCF dataset.

    gt_percentage = [100. * val / sum(gt_severity_dict.values()) for val in gt_severity_dict.values()]
    sim_percentage = [100. * val / sum(sim_severity_dict.values()) if sum(sim_severity_dict.values()) != 0 else np.nan for val in sim_severity_dict.values()]

    fig, ax = plt.subplots(figsize=(16, 9))
    fontsize = 40
    plt.rcParams['font.size'] = str(fontsize)
    plt.rcParams["font.family"] = "Arial"

    x = np.array([0 + 1.2 * i for i in range(len(severity_type))])
    bar_width = 0.3
    ax.bar(x - bar_width / 2, gt_percentage, width=bar_width, color='C0', align='center', label='Ground truth')
    ax.bar(x + bar_width / 2, sim_percentage, width=bar_width, color='C1', align='center', label='NeuralNDE')

    ax.set_xticks([0 + 1.2 * i for i in range(len(severity_type))])
    xticks = severity_type
    ax.set_xticklabels(xticks, fontsize=fontsize)
    # plt.xticks(rotation=45, fontsize=fontsize)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Probability (%)', fontsize=fontsize)
    plt.ylim(0, 100)
    plt.legend(fontsize=fontsize)

    H_dist = cal_Hellinger_dis(P=gt_percentage, Q=sim_percentage)
    KL_div = cal_KL_div(P=gt_percentage, Q=sim_percentage)

    # Data
    res_df = pd.DataFrame([gt_percentage, sim_percentage], index=["Ground_truth", "NeuralNDE"], columns=severity_type)
    save_path = os.path.join(output_folder, 'crash_severity.csv')
    res_df.to_csv(save_path, index=True)

    # Save figure with x, y axis label and ticks
    output_folder_with_label = os.path.join(output_folder)
    os.makedirs(output_folder_with_label, exist_ok=True)
    save_file_name = 'crash_severity_Hdis{0}_KL{1}'.format("%.3f" % round(H_dist, 3), "%.3f" % round(KL_div, 3))
    plt.savefig(os.path.join(output_folder_with_label, save_file_name + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder_with_label, save_file_name + '.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder_with_label, save_file_name + '.svg'), bbox_inches='tight')
    plt.close()


# Quantify the difference of the distribution
def cal_Hellinger_dis(P, Q):
    """
    Calculate the Hellinger distance for two distribution
    """
    # Convert into probability first
    P = P/np.sum(P)
    Q = Q/np.sum(Q)

    H_dist = (1 / np.sqrt(2)) * np.linalg.norm(np.abs(np.sqrt(P) - np.sqrt(Q)))
    return H_dist


def cal_KL_div(P, Q):
    """
    Calculate the KL-divergence for two distribution
    sum(P(x) * log (P(x)/Q(x)))
    """
    # Convert into probability first
    P = P/np.sum(P)
    Q = Q/np.sum(Q)

    KL_div = entropy(pk=P, qk=Q, base=None, axis=0)
    return KL_div


if __name__ == '__main__':
    # Simulation results
    experiment_name = 'AA_rdbt_paper_results'
    res_folder = f'../data/paper-inference-results/{experiment_name}'
    output_folder = os.path.join(f'plot/{experiment_name}/crash_severity')
    os.makedirs(output_folder, exist_ok=True)

    # Analyze the results
    analyze_crash_severity(res_folder, output_folder)