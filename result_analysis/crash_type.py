"""
This file is to plot crash type results.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
from shapely.geometry import Polygon, Point
from itertools import combinations
from scipy.stats import entropy


def collision_check(vehicle_list, extra_buffer=False):
    """
    Check whether collision happens
    """
    collision_flag = False
    collision_vehicle_list = []

    for vehicle_pair in combinations(vehicle_list, r=2):
        v1, v2 = vehicle_pair[0], vehicle_pair[1]
        if extra_buffer:
            v1_poly, v2_poly = v1.safe_poly_box, v2.safe_poly_box
        else:
            v1_poly, v2_poly = v1.poly_box, v2.poly_box

        if v1_poly.intersects(v2_poly):
            collision_flag = True
            if v1 not in collision_vehicle_list:
                collision_vehicle_list.append(v1)
            if v2 not in collision_vehicle_list:
                collision_vehicle_list.append(v2)

    return collision_flag, collision_vehicle_list


def _get_region_coordinates(anchor_center_pt, direction_pt1, direction_pt2, distance=100):
    direction1 = np.arctan2(direction_pt1[1] - anchor_center_pt[1], direction_pt1[0] - anchor_center_pt[0])
    new_pt1 = [anchor_center_pt[0] + distance * np.cos(direction1), anchor_center_pt[1] + distance * np.sin(direction1)]

    direction2 = np.arctan2(direction_pt2[1] - anchor_center_pt[1], direction_pt2[0] - anchor_center_pt[0])
    new_pt2 = [anchor_center_pt[0] + distance * np.cos(direction2), anchor_center_pt[1] + distance * np.sin(direction2)]

    return anchor_center_pt, new_pt1, new_pt2


def relative_position_two_vehicles(anchor_v, determine_v):
    anchor_center_pt = [anchor_v.location.x, anchor_v.location.y]
    pt1, pt2, pt3, pt4 = anchor_v.realworld_4_vertices  # upper left clockwise

    determine_v_center = Point(determine_v.location.x, determine_v.location.y)

    relative_position = None
    # front
    front_area = Polygon(_get_region_coordinates(anchor_center_pt, pt1, pt2, distance=100))
    if front_area.contains(determine_v_center):
        relative_position = 'front'
        return relative_position

    # rear
    rear_area = Polygon(_get_region_coordinates(anchor_center_pt, pt3, pt4, distance=100))
    if rear_area.contains(determine_v_center):
        relative_position = 'rear'
        return relative_position

    # left
    left_area = Polygon(_get_region_coordinates(anchor_center_pt, pt1, pt4, distance=100))
    if left_area.contains(determine_v_center):
        relative_position = 'left'
        return relative_position

    # right
    right_area = Polygon(_get_region_coordinates(anchor_center_pt, pt2, pt3, distance=100))
    if right_area.contains(determine_v_center):
        relative_position = 'right'
        return relative_position


def relative_heading_two_vehicles(anchor_v, determine_v):
    """
    Return the relative angle between two vehicles heading.
    Return in degrees in [0, 180].
    """
    angle = abs(anchor_v.speed_heading - determine_v.speed_heading) % 360
    relative_heading = angle if (0 <= angle <= 180) else 360. - angle
    assert (0. <= relative_heading <= 180.)

    return relative_heading


def _collision_type(relative_position, relative_heading):
    collision_type = 'other'
    if relative_position == 'rear' and relative_heading <= 40.:
        collision_type = 'rear_end'
    if relative_position == 'rear' and 40. < relative_heading < 90.:
        collision_type = 'angle'

    if relative_position == 'front' and relative_heading < 40.:
        collision_type = 'rear_end'
    if relative_position == 'front' and 40. < relative_heading <= 90.:
        collision_type = 'angle'
    if relative_position == 'front' and 90. < relative_heading:
        # collision_type = 'head_on'
        # currently classify head_on crash as other since there is limited number of head on crash in roundabout
        collision_type = 'other'

    if (relative_position == 'left' or relative_position == 'right') and ((relative_heading < 30.) or (relative_heading > 150.)):
        collision_type = 'sideswipe'
    if (relative_position == 'left' or relative_position == 'right') and (30. < relative_heading < 150.):
        collision_type = 'angle'

    return collision_type


def collision_type_classification(collision_vehicle_list):
    """
    Do collision type classification based on relative position and heading of colliding vehicles.
    Currently, we only consider two vehicle collision case.
    """
    collision_type = 'other'
    relative_position, relative_heading = None, None
    if len(collision_vehicle_list) == 2:
        v1, v2 = collision_vehicle_list
        relative_position = relative_position_two_vehicles(anchor_v=v1, determine_v=v2)
        relative_heading = relative_heading_two_vehicles(anchor_v=v1, determine_v=v2)
        collision_type = _collision_type(relative_position, relative_heading)
    if len(collision_vehicle_list) > 2:
        collision_type = 'more_than_two_vehicles'
    return collision_type, relative_position, relative_heading


def analyze_collision_type_distribution(res_folder, output_folder, plot_collision_type=['rear_end', 'angle', 'sideswipe']):

    # Ground-truth
    gt_collision_type_distribution = {"rear_end": 0.133, "angle": 0.546, "sideswipe": 0.321, "other": 0.0, "more_than_two_vehicles": 0.0}  # Data from 2016-2020 crashes MTCF.

    collision_type_num = {'rear_end': 0, 'angle': 0, 'sideswipe': 0, 'other': 0, 'more_than_two_vehicles': 0}

    traj_dirs = sorted(glob.glob(os.path.join(res_folder, '*/*/*.pickle')))
    traj_dirs_df = pd.DataFrame(traj_dirs, columns=['path'])
    traj_dirs_df['folder'] = traj_dirs_df["path"].apply(lambda x: x.split('\\')[-3])
    traj_dirs_df['episode_frame'] = traj_dirs_df["path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    traj_dirs_df["episode_id"] = (traj_dirs_df['episode_frame'].apply(lambda x: x.split('-')[0])).astype(int)
    traj_dirs_df["frame_id"] = (traj_dirs_df['episode_frame'].apply(lambda x: x.split('-')[1])).astype(int)

    for folder_id in traj_dirs_df.folder.unique():
        one_folder_data = traj_dirs_df[traj_dirs_df['folder'] == folder_id]
        for episode_id in one_folder_data.episode_id.unique():
            one_episode = one_folder_data[one_folder_data['episode_id'] == episode_id]
            collision_frame_path = one_episode[one_episode.frame_id == one_episode.frame_id.max()]['path'].values[0]
            vehicle_list = pickle.load(open(collision_frame_path, "rb"))
            collision_flag, collision_vehicle_list = collision_check(vehicle_list=vehicle_list)
            if not collision_flag:
                continue
            collision_type, _, _ = collision_type_classification(collision_vehicle_list)
            collision_type_num[collision_type] += 1

    keys = collision_type_num.keys()
    values = [collision_type_num[key] / sum(collision_type_num.values()) if sum(collision_type_num.values()) != 0 else np.nan for key in collision_type_num.keys()]
    collision_type_distribution = dict(zip(keys, values))

    plot_collision_type_distribution_comparison(gt_collision_type_distribution, collision_type_distribution, output_folder, plot_collision_type=plot_collision_type)

    res_df = pd.DataFrame.from_records([{k: v*100 for k, v in gt_collision_type_distribution.items()}, {k: v*100 for k, v in collision_type_distribution.items()}]).fillna(0)
    res_df = res_df.drop(columns=['other', 'more_than_two_vehicles'])
    res_df.index = ['Ground_truth', 'NeuralNDE']

    save_path = os.path.join(output_folder, 'collision_type_results.csv')
    res_df.to_csv(save_path, index=True)


def plot_collision_type_distribution_comparison(gt_collision_type_distribution, sim_collision_type_distribution, output_folder, plot_collision_type=['rear_end', 'angle', 'sideswipe']):
    gt_res = [100 * gt_collision_type_distribution[key] for key in plot_collision_type]
    sim_res = [100 * sim_collision_type_distribution[key] for key in plot_collision_type]

    fig, ax = plt.subplots(figsize=(16, 9))
    fontsize = 40
    plt.rcParams['font.size'] = str(fontsize)
    plt.rcParams["font.family"] = "Arial"

    x = np.array([0 + 1.0 * i for i in range(len(plot_collision_type))])
    bar_width = 0.25
    ax.bar(x - bar_width / 2, gt_res, width=bar_width, color='C0', align='center', label='Ground truth')
    ax.bar(x + bar_width / 2, sim_res, width=bar_width, color='C1', align='center', label='NeuralNDE')

    ax.set_xticks([0 + 1.0 * i for i in range(len(plot_collision_type))])
    xticks_mapping = {'rear_end': 'Rear end', 'angle': 'Angle', 'sideswipe': 'Sideswipe', 'other': 'Other', 'more_than_two_vehicles': 'More than two vehicles'}
    xticks = [xticks_mapping[key] for key in plot_collision_type]
    ax.set_xticklabels(xticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Probability (%)', fontsize=fontsize)
    plt.ylim(0, 100)
    plt.legend(fontsize=fontsize)

    H_dist = cal_Hellinger_dis(P=gt_res, Q=sim_res)
    KL_div = cal_KL_div(P=gt_res, Q=sim_res)

    # Save figure with x, y axis label and ticks
    output_folder_with_label = os.path.join(output_folder)
    os.makedirs(output_folder_with_label, exist_ok=True)
    save_file_name = 'collision_type_distribution_comparison_Hdis{0}_KL{1}'.format("%.3f" % round(H_dist, 3), "%.3f" % round(KL_div, 3))
    plt.savefig(os.path.join(output_folder_with_label, save_file_name + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder_with_label, save_file_name + '.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder_with_label, save_file_name + '.pdf'), bbox_inches='tight')


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
    res_folder = r'./raw_data/NeuralNDE/'
    output_folder = os.path.join('plot/crash_type')
    os.makedirs(output_folder, exist_ok=True)

    # Analyze the results
    analyze_collision_type_distribution(res_folder, output_folder, plot_collision_type=['rear_end', 'angle', 'sideswipe'])