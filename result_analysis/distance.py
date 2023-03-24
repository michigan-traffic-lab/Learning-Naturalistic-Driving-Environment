"""
This file is to plot vehicle distance distribution.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy


def analyze_distance_res(res_folder, output_folder, density=True, gt_folder=None, save_fig=False):
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
    distance_list = []
    for tmp in output_distance_list:
        distance_list += tmp

    # plt.figure(figsize=(16, 9))
    plt.figure(figsize=(12, 12))
    fontsize = 40
    plt.rcParams['font.size'] = str(fontsize)
    plt.rcParams['font.family'] = 'Arial'
    lb, ub, num = 0, 80, 161
    bins = np.linspace(lb, ub, num)

    gt_prob_density, gt_bins, _ = plt.hist(gt_distance_list, bins, density=density, label='Ground truth', alpha=0.5, edgecolor='k')
    sim_prob_density, sim_bins, _ = plt.hist(distance_list, bins, density=density, label='NeuralNDE', alpha=0.5, edgecolor='k')

    H_dist = cal_Hellinger_dis(P=gt_prob_density, Q=sim_prob_density)
    KL_div = cal_KL_div(P=gt_prob_density, Q=sim_prob_density)

    plt.legend()
    plt.xlabel('Distance (m)')
    plt.ylim(0, 0.035)
    plt.ylabel('Probability density' if density else 'frequency')
    plt.yticks([0, 0.01, 0.02, 0.03])

    data = pd.DataFrame({'bins': gt_bins[1:], 'ground_truth_probability_density': gt_prob_density, f'NeuralNDE_probability_density': sim_prob_density})
    bin_size = bins[1] - bins[0]
    gt_probability_all = np.sum(gt_prob_density * bin_size)
    sim_probability_all = np.sum(sim_prob_density * bin_size)
    print(f"Sum gt probability: {gt_probability_all}, sum sim probability: {sim_probability_all}")
    print(f"Hellinger distance: {H_dist}, KL divergence: {KL_div}")

    # Save figure with x, y axis label and ticks
    if save_fig:
        output_folder_with_label = os.path.join(output_folder, 'distance')
        os.makedirs(output_folder_with_label, exist_ok=True)
        file_name = 'distance_distribution_Hdis{0}_KL{1}'.format("%.3f" % round(H_dist, 3), "%.3f" % round(KL_div, 3))
        plt.savefig(os.path.join(output_folder_with_label, file_name + '.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder_with_label, file_name + '.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(output_folder_with_label, file_name + '.pdf'), bbox_inches='tight')
        data.to_csv(os.path.join(output_folder_with_label, file_name + '.csv'), sep=",", index=False)


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
    # Ground-truth results
    gt_folder = r'./raw_data/ground_truth/'

    # Simulation results
    res_folder = r'./raw_data/NeuralNDE/'
    save_fig = True  # Whether save plotted figure
    output_folder = os.path.join('plot')

    # Analyze the results and plot the figures
    analyze_distance_res(res_folder, output_folder, density=True, gt_folder=gt_folder, save_fig=save_fig)
