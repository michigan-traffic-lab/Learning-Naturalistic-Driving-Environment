"""
This file is to plot vehicle yielding distance and yielding speed distributions.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy


def combine_dict(dict_list, nested_dict=False):
    """
    Combine a list dicts with same keys and sum them up into a single dict
    """
    if nested_dict:
        new_dict = {}
        for key1 in dict_list[0].keys():
            new_dict[key1] = {}
            for key2 in dict_list[0][key1].keys():
                val = 0
                for d in dict_list:
                    val += d[key1][key2]
                new_dict[key1][key2] = val
    else:
        c = Counter()
        for d in dict_list:
            c.update(d)
        new_dict = dict(c)
    return new_dict


def analyze_yielding_dist_and_v_res(res_folder, output_folder, gt_folder, density=True):
    output_yielding_conflict_dist_and_v_dict_list = []
    for subfolder in sorted(os.listdir(res_folder)):
        try:
            with open(os.path.join(res_folder, subfolder, "output_yielding_conflict_dist_and_v_dict_list.json")) as json_file:
                output_yielding_conflict_dist_and_v_dict_list += json.load(json_file)
        except:
            pass

    if len(output_yielding_conflict_dist_and_v_dict_list) == 0:
        print("============== No yielding results found, check folder path and "
              "whether gen_realistic_metric_flag is set to True in the config file"
              "when running the simulation.")
        return
    yielding_conflict_dist_and_v_dict = combine_dict(output_yielding_conflict_dist_and_v_dict_list, nested_dict=False)
    yield_dist_and_v_list, not_yield_dist_and_v_list = yielding_conflict_dist_and_v_dict['yield_dist_and_v_list'], yielding_conflict_dist_and_v_dict['not_yield_dist_and_v_list']

    yield_dist, yield_v = np.array(yield_dist_and_v_list)[:, 0].tolist(), np.array(yield_dist_and_v_list)[:, 1].tolist()

    if gt_folder is not None:
        gt_yielding_conflict_dist_and_v_dict_list = []
        try:
            with open(os.path.join(gt_folder, "output_yielding_conflict_dist_and_v_dict_list.json")) as json_file:
                gt_yielding_conflict_dist_and_v_dict_list += json.load(json_file)
        except:
            ValueError("No file: {0}".format(os.path.join(gt_folder, "output_yielding_conflict_dist_and_v_dict_list.json")))
        gt_yielding_conflict_dist_and_v_dict = combine_dict(gt_yielding_conflict_dist_and_v_dict_list, nested_dict=False)
        gt_yield_dist_and_v_list, gt_not_yield_dist_and_v_list = gt_yielding_conflict_dist_and_v_dict['yield_dist_and_v_list'], gt_yielding_conflict_dist_and_v_dict[
            'not_yield_dist_and_v_list']

        gt_yield_dist, gt_yield_v = np.array(gt_yield_dist_and_v_list)[:, 0].tolist(), np.array(gt_yield_dist_and_v_list)[:, 1].tolist()
        gt_yield_dist = [val for val in gt_yield_dist if val < 45]

    plot_yield_dist_distribution(yield_dist, gt_yield_dist, output_folder, density=density)
    plot_yield_speed_distribution(yield_v, gt_yield_v, output_folder, density=density)


def plot_yield_dist_distribution(yield_dist, gt_yield_dist, output_folder, density=False):
    plt.figure(figsize=(12, 12))
    fontsize = 40
    plt.rcParams['font.size'] = str(fontsize)
    plt.rcParams["font.family"] = "Arial"
    bins = np.linspace(0, 50, 150)
    gt_prob_density, gt_bins, _ = plt.hist(gt_yield_dist, bins, density=density, label='Ground truth', alpha=0.5, edgecolor='k')
    sim_prob_density, _, _ = plt.hist(yield_dist, bins, density=density, label='NeuralNDE', alpha=0.5, edgecolor='k')

    H_dist = cal_Hellinger_dis(P=gt_prob_density, Q=sim_prob_density)
    KL_div = cal_KL_div(P=gt_prob_density, Q=sim_prob_density)

    plt.ylim(0, 0.11)
    plt.yticks([0, 0.03, 0.06, 0.09])
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.legend(fontsize=fontsize)
    plt.xlabel('Yielding distance (m)')
    plt.ylabel('Probability density' if density else 'Frequency')

    data = pd.DataFrame({'bins': gt_bins[1:], 'ground_truth_probability_density': gt_prob_density, f'NeuralNDE_probability_density': sim_prob_density})
    bin_size = bins[1] - bins[0]
    gt_probability_all = np.sum(gt_prob_density * bin_size)
    sim_probability_all = np.sum(sim_prob_density * bin_size)
    print(f"Yielding distance sum gt probability: {gt_probability_all}, sum sim probability: {sim_probability_all}")
    print(f"Yielding distance Hellinger distance: {H_dist}, KL divergence: {KL_div}")

    # Save figure with x, y axis label and ticks
    file_name = 'Y_dist_distribution_Hdis{0}_KL{1}'.format("%.3f" % round(H_dist, 3), "%.3f" % round(KL_div, 3))
    plt.savefig(os.path.join(output_folder, file_name + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, file_name + '.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, file_name + '.pdf'), bbox_inches='tight')
    data.to_csv(os.path.join(output_folder, file_name + '.csv'), sep=",", index=False)


def plot_yield_speed_distribution(yield_v, gt_yield_v, output_folder, density=False):
    plt.figure(figsize=(12, 12))
    fontsize = 40
    plt.rcParams['font.size'] = str(fontsize)
    plt.rcParams["font.family"] = "Arial"
    bins = np.linspace(0, 15.2, 100)
    gt_prob_density, gt_bins, _ = plt.hist(gt_yield_v, bins, density=density, label='Ground truth', alpha=0.5, edgecolor='k')
    sim_prob_density, sim_bins, _ = plt.hist(yield_v, bins, density=density, label='NeuralNDE', alpha=0.5, edgecolor='k')

    H_dist = cal_Hellinger_dis(P=gt_prob_density, Q=sim_prob_density)
    KL_div = cal_KL_div(P=gt_prob_density, Q=sim_prob_density)
    plt.ylim(0, 0.35)
    plt.yticks([0, 0.1, 0.2, 0.3])
    plt.xticks([0, 3, 6, 9, 12, 15])
    plt.legend(fontsize=fontsize)
    plt.xlabel('Yielding speed (m/s)')
    plt.ylabel('Probability density' if density else 'Frequency')

    data = pd.DataFrame({'bins': gt_bins[1:], 'ground_truth_probability_density': gt_prob_density, f'NeuralNDE_probability_density': sim_prob_density})
    bin_size = bins[1] - bins[0]
    gt_probability_all = np.sum(gt_prob_density * bin_size)
    sim_probability_all = np.sum(sim_prob_density * bin_size)
    print(f"Yielding speed sum gt probability: {gt_probability_all}, sum sim probability: {sim_probability_all}")
    print(f"Yielding speed Hellinger distance: {H_dist}, KL divergence: {KL_div}")

    # Save figure with x, y axis label and ticks
    file_name = 'Y_v_distribution_Hdis{0}_KL{1}'.format("%.3f" % round(H_dist, 3), "%.3f" % round(KL_div, 3))
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
    # Simulation results. Change this to the folder of your results
    location = 'AA_rdbt'  # 'AA_rdbt' or 'rounD'
    experiment_name = f'{location}_paper_results'
    res_folder = f'../data/paper-inference-results/{experiment_name}'
    # Ground-truth results
    if location == "AA_rdbt":
        gt_folder = r'../data/statistical-realism-ground-truth/AA_rdbt_ground_truth/'
    elif location == 'rounD':
        gt_folder = r'../data/statistical-realism-ground-truth/rounD_ground_truth/'
    else:
        raise NotImplementedError(
            '{0} does not supported yet...Choose from ["AA_rdbt", "rounD"].'.format(location))

    output_folder = os.path.join(f'plot/{experiment_name}/yielding_distance_and_speed')
    os.makedirs(output_folder, exist_ok=True)

    # Analyze the results and plot the figures
    analyze_yielding_dist_and_v_res(res_folder, output_folder, gt_folder=gt_folder, density=True)
