"""
This file is to plot vehicle distance distribution in near-miss situations.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bisect
from scipy.stats import entropy


def analyze_distance_in_near_miss_res(res_folder, output_folder, dis_threshold=10.0, density=True, gt_folder=None):
    if gt_folder is not None:
        gt_distance_list = []
        try:
            with open(os.path.join(gt_folder, "output_distance_list_three_circle.json")) as json_file:
                gt_distance_list += json.load(json_file)
        except:
            raise ValueError("No file: {0}".format(os.path.join(gt_folder, "output_distance_list_three_circle.json")))

        gt_distance_list = sum(gt_distance_list, [])

    output_distance_list = []
    for subfolder in sorted(os.listdir(res_folder)):
        try:
            file_name = os.path.join(res_folder, subfolder, "output_distance_list_three_circle.json")
            with open(file_name) as json_file:
                output_distance_list += json.load(json_file)
        except:
            pass

    if len(output_distance_list) == 0:
        print("============== No distance results found, check folder path and "
              "whether gen_realistic_metric_flag is set to True in the config file"
              "when running the simulation.")

    distance_list = []
    for tmp in output_distance_list:
        distance_list += tmp

    plt.figure(figsize=(16, 9))
    fontsize = 40
    plt.rcParams['font.size'] = str(fontsize)
    plt.rcParams["font.family"] = "Arial"
    lb, ub, num = 0, dis_threshold, 2*dis_threshold + 1
    bins = np.linspace(lb, ub, num)
    gt_prob_density, gt_bins, _ = plt.hist(gt_distance_list, bins, density=density, label='Ground truth', alpha=0.5, edgecolor='k')
    sim_prob_density, sim_bins, _ = plt.hist(distance_list, bins, density=density, label='NeuralNDE', alpha=0.5, edgecolor='k')

    H_dist = cal_Hellinger_dis(P=gt_prob_density, Q=sim_prob_density)
    KL_div = cal_KL_div(P=gt_prob_density, Q=sim_prob_density)

    plt.legend()
    plt.xlabel('Distance (m)')
    if dis_threshold:
        plt.xlim(-1, dis_threshold)
    plt.ylim(0, 0.3)
    plt.ylabel('Probability density' if density else 'frequency')
    plt.yticks([0, 0.1, 0.2, 0.3])

    # Find the index of a value with left value smaller than threshold and right value greater than threshold
    index = bisect.bisect_left(bins, dis_threshold)
    data = pd.DataFrame({'bins': gt_bins[1:index+1], 'ground_truth_probability_density': gt_prob_density[:index], f'NeuralNDE_probability_density': sim_prob_density[:index]})

    # Save figure with x, y axis label and ticks
    file_name = 'distance_in_near_miss_distribution_Hdis{0}_KL{1}'.format("%.3f" % round(H_dist, 3), "%.3f" % round(KL_div, 3))
    plt.savefig(os.path.join(output_folder, file_name + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, file_name + '.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, file_name + '.pdf'), bbox_inches='tight')
    data.to_csv(os.path.join(output_folder, file_name + '.csv'), sep=",", index=False)


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
    output_folder = os.path.join(f'plot/{experiment_name}/distance_in_near_miss')
    os.makedirs(output_folder, exist_ok=True)

    # Ground-truth results
    gt_folder = r'../data/statistical-realism-ground-truth/AA_rdbt_ground_truth/'

    # Analyze the results and plot the figures
    analyze_distance_in_near_miss_res(res_folder, output_folder, dis_threshold=10, density=True, gt_folder=gt_folder)
