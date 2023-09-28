import os

from crash_rate import analyze_collision_rate
from crash_severity import analyze_crash_severity
from crash_type import analyze_collision_type_distribution
from distance import analyze_distance_res
from distance_in_near_miss import analyze_distance_in_near_miss_res
from instantaneous_speed import analyze_instant_speed_res
from PET import analyze_PET_res
from yielding_distance_and_speed import analyze_yielding_dist_and_v_res

if __name__ == '__main__':
    # Simulation results. Change this to the folder of your results
    location = 'AA_rdbt'  # 'AA_rdbt' or 'rounD'
    experiment_name = 'NNDE-open-source-rerun-AA-rdbt-09122023-tune-crash-rate-5'  # 'AA_rdbt_paper_results' or 'rounD_paper_results' or your experiment
    # res_folder = f'../data/paper-inference-results/{experiment_name}'
    res_folder = f'D:/NNDE-open-source/results\inference/AA_rdbt_inference/{experiment_name}/3600s/raw_data'

    # Ground-truth results
    if location == "AA_rdbt":
        gt_folder = r'../data/statistical-realism-ground-truth/AA_rdbt_ground_truth/'
    elif location == 'rounD':
        gt_folder = r'../data/statistical-realism-ground-truth/rounD_ground_truth/'
    else:
        raise NotImplementedError(
            '{0} does not supported yet...Choose from ["AA_rdbt", "rounD"].'.format(location))

    # Instantaneous speed
    print("Analyzing instantaneous speed...")
    output_folder = os.path.join(f'plot/{experiment_name}/instantaneous_speed')
    os.makedirs(output_folder, exist_ok=True)
    analyze_instant_speed_res(res_folder, output_folder, density=True, gt_folder=gt_folder)
    print('Done.\n')

    # Distance
    print("Analyzing distance...")
    output_folder = os.path.join(f'plot/{experiment_name}/distance')
    os.makedirs(output_folder, exist_ok=True)
    analyze_distance_res(res_folder, output_folder, density=True, gt_folder=gt_folder)
    print('Done.\n')

    # Yielding distance and speed
    print("Analyzing yielding distance and speed...")
    output_folder = os.path.join(f'plot/{experiment_name}/yielding_distance_and_speed')
    os.makedirs(output_folder, exist_ok=True)
    analyze_yielding_dist_and_v_res(res_folder, output_folder, gt_folder=gt_folder, density=True)
    print('Done.\n')

    # Plot safety-critical statistics for AA_rdbt since there is no rounD ground-truth
    if location == "AA_rdbt":
        # Crash rate
        print("Analyzing crash rate...")
        output_folder = os.path.join(f'plot/{experiment_name}/crash_rate')
        os.makedirs(output_folder, exist_ok=True)
        analyze_collision_rate(res_folder, output_folder)
        print('Done.\n')

        # Crash severity
        print("Analyzing crash severity...")
        output_folder = os.path.join(f'plot/{experiment_name}/crash_severity')
        os.makedirs(output_folder, exist_ok=True)
        analyze_crash_severity(res_folder, output_folder)
        print('Done.\n')

        # Crash type
        print("Analyzing crash type...")
        output_folder = os.path.join(f'plot/{experiment_name}/crash_type')
        os.makedirs(output_folder, exist_ok=True)
        analyze_collision_type_distribution(res_folder, output_folder, plot_collision_type=['rear_end', 'angle', 'sideswipe'])
        print('Done.\n')

        # Distance in near-miss
        print("Analyzing distance in near-miss...")
        output_folder = os.path.join(f'plot/{experiment_name}/distance_in_near_miss')
        os.makedirs(output_folder, exist_ok=True)
        analyze_distance_in_near_miss_res(res_folder, output_folder, density=True, gt_folder=gt_folder,
                                          dis_threshold=10)
        print('Done.\n')

        # PET
        print("Analyzing PET...")
        output_folder = os.path.join(f'plot/{experiment_name}/PET')
        os.makedirs(output_folder, exist_ok=True)
        analyze_PET_res(res_folder, output_folder, density=True, gt_folder=gt_folder, sim_resol=0.4, PET_threshold=4)
        print('Done.\n')
