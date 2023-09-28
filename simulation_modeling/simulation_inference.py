import os
import glob
import random
import pickle
import numpy as np
import cv2
import torch
import json
import pandas as pd
import time

from simulation_modeling.traffic_simulator import TrafficSimulator
from simulation_modeling.crashcritic import CrashCritic
from simulation_modeling.trajectory_interpolator import TrajInterpolator
from simulation_modeling.vehicle_generator import AA_rdbt_TrafficGenerator, rounD_TrafficGenerator
from sim_evaluation_metric.realistic_metric import RealisticMetrics

from basemap import Basemap

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimulationInference(object):
    def __init__(self, configs):

        self.dataset = configs["dataset"]
        self.history_length, self.pred_length, self.m_tokens = configs["history_length"], configs["pred_length"], configs["m_tokens"]
        self.rolling_step = configs["rolling_step"]
        self.sim_wall_time = configs["sim_wall_time"]
        self.sim_resol = configs["sim_resol"]
        self.use_neural_safety_mapping = configs["use_neural_safety_mapping"]
        self.background_map = Basemap(map_file_dir=configs["basemap_dir"], map_height=configs["map_height"], map_width=configs["map_width"])
        self.viz_flag = configs["viz_flag"]
        self.save_viz_flag = configs["save_viz_flag"]
        self.save_collision_data_flag = configs["save_collision_data_flag"]
        self.save_fps = configs["save_fps"]
        self.save_viz_folder = configs["save_viz_folder"]
        self.device = configs["device"]

        if configs["dataset"] == "AA_rdbt":
            self.traffic_generator = AA_rdbt_TrafficGenerator(config=configs["traffic_generator_config"])
        elif configs["dataset"] == 'rounD':
            self.traffic_generator = rounD_TrafficGenerator(config=configs["traffic_generator_config"])
        else:
            raise NotImplementedError('{0} does not supported yet...Choose from ["AA_rdbt", "rounD"].'.format(configs["dataset"]))

        self.max_m_steps = int((self.sim_wall_time / self.sim_resol) / self.rolling_step)
        self.traj_dirs, self.subfolder_data_proportion, self.subsubfolder_data_proportion = self._get_traj_dirs(path_to_traj_data=configs["init_traj_clips_dir"])

        self.sim = TrafficSimulator(model=configs["model"], history_length=self.history_length, pred_length=self.pred_length, m_tokens=self.m_tokens,
                                    checkpoint_dir=configs["behavior_model_ckpt_dir"],
                                    safety_mapper_ckpt_dir=configs["safety_mapper_ckpt_dir"], drivable_map_dir=configs["drivable_map_dir"], device=self.device,
                                    sim_remove_vehicle_area_map=configs["sim_remove_vehicle_area_map"], map_height=configs["map_height"], map_width=configs["map_width"])

        # Basic simulation results of all episodes, units: [episodes], [sim steps], [m], [s]
        # Will always be saved regardless of realistic_metric (e.g., vehicle speed, distance, etc.)
        self.colli_num, self.total_sim_steps, self.total_travel_distances, self.total_sim_wall_time = 0, 0, 0, 0
        # Basic simulation stats of one simulation episode (the current simulating episode).
        self.one_sim_TIME_BUFF, self.one_sim_steps_all_vehicles, self.one_sim_travel_distances_all_vehicles = None, None, None
        self.one_sim_wall_time, self.one_sim_colli_flag, self.one_sim_TIME_BUFF_last_moment, self.one_sim_TIME_BUFF_newly_generated, self.one_sim_whole_sim_do_safety_mapping_flag = 0., False, None, None, True

        self.gen_realistic_metric_flag = configs["gen_realistic_metric_flag"]  # Whether to generate metrics.
        self.gen_realistic_metric_dict = configs["gen_realistic_metric_dict"]  # What metrics to generate.
        self.realistic_metric_save_folder = configs["realistic_metric_save_folder"]
        os.makedirs(self.realistic_metric_save_folder, exist_ok=True)

        if self.gen_realistic_metric_flag:

            # ROIs
            circle_map_dir = os.path.join(configs["ROI_map_dir"], 'circle')
            entrance_map_dir = os.path.join(configs["ROI_map_dir"], 'entrance')
            exit_map_dir = os.path.join(configs["ROI_map_dir"], 'exit')
            crosswalk_map_dir = None  # os.path.join(configs["ROI_map_dir"], 'crosswalk')
            yielding_area_map_dir = os.path.join(configs["ROI_map_dir"], 'yielding-area')
            at_circle_lane_map_dir = os.path.join(configs["ROI_map_dir"], 'at-circle-lane')

            # PET (post-encroachment time) analysis configs
            if self.dataset == 'AA_rdbt':
                basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
                basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
                basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
                basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

                PET_configs = configs["PET_configs"]
                PET_configs["basemap_img"] = basemap_img

            elif self.dataset == 'rounD':
                basemap_img = cv2.imread(configs["basemap_dir"], cv2.IMREAD_COLOR)
                basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_BGR2RGB)
                basemap_img = cv2.resize(basemap_img, (configs["map_width"], configs["map_height"]))
                basemap_img = (basemap_img.astype(np.float64) * 0.6).astype(np.uint8)

                PET_configs = configs["PET_configs"]
                PET_configs["basemap_img"] = basemap_img

            self.SimMetricsAnalyzer = RealisticMetrics(drivable_map_dir=configs["drivable_map_dir"], sim_remove_vehicle_area_map=configs["sim_remove_vehicle_area_map"],
                                                       circle_map_dir=circle_map_dir, entrance_map_dir=entrance_map_dir, exit_map_dir=exit_map_dir,
                                                       crosswalk_map_dir=crosswalk_map_dir, yielding_area_map_dir=yielding_area_map_dir, at_circle_lane_map_dir=at_circle_lane_map_dir,
                                                       sim_resol=self.sim_resol,
                                                       map_height=configs["map_height"], map_width=configs["map_width"],
                                                       PET_configs=PET_configs)

            self.output_instant_speed_list = []  # This list is instant speed in the circle
            self.output_yielding_conflict_dist_and_v_dict_list = []
            self.output_distance_all_vehicle_pairs_list_three_circle = []
            self.output_PET_list = []  # Post-encroachment time results.

        self.save_simulated_TIME_BUFF_flag = configs["save_simulated_TIME_BUFF_flag"]
        self.simulated_TIME_BUFF_save_folder = configs["simulated_TIME_BUFF_save_folder"]

        # initialize conflict critic
        self.use_conflict_critic_module = configs["use_conflict_critic_module"]
        if self.use_conflict_critic_module:
            self.conflict_critic_agent = CrashCritic(sim_resol=self.sim_resol)

        self.interpolate_flag = configs["interpolate_flag"]
        if self.interpolate_flag:
            # initialize the trajectory interpolator
            # The number of steps interpolate between predicted steps. For example, if resolution is 0.4s and steps is 3, then new resolution is 0.1s.
            self.intep_steps = configs["intep_steps"]
            self.traj_interpolator = TrajInterpolator()

    @staticmethod
    def _get_traj_dirs(path_to_traj_data):
        subfolders = sorted(os.listdir(os.path.join(path_to_traj_data)))
        subsubfolders = [sorted(os.listdir(os.path.join(path_to_traj_data, subfolders[i]))) for i in
                         range(len(subfolders))]

        traj_dirs = []
        each_subfolder_size = []
        each_subsubfolder_size = []
        for i in range(len(subfolders)):
            one_video = []
            each_subsubfolder_size_tmp = []
            for j in range(len(subsubfolders[i])):
                files_list = sorted(glob.glob(os.path.join(path_to_traj_data, subfolders[i], subsubfolders[i][j], '*.pickle')))
                one_video.append(files_list)
                each_subsubfolder_size_tmp.append(len(files_list))
            traj_dirs.append(one_video)
            each_subfolder_size.append(sum([len(listElem) for listElem in one_video]))
            each_subsubfolder_size.append(each_subsubfolder_size_tmp)

        subfolder_data_proportion = [each_subfolder_size[i] / sum(each_subfolder_size) for i in range(len(each_subfolder_size))]
        subsubfolder_data_proportion = [[each_subsubfolder_size[i][j] / sum(each_subsubfolder_size[i]) for j in range(len(each_subsubfolder_size[i]))] for i in
                                        range(len(each_subsubfolder_size))]

        return traj_dirs, subfolder_data_proportion, subsubfolder_data_proportion

    def _initialize_sim(self):
        subfolder_id = random.choices(range(len(self.subfolder_data_proportion)), weights=self.subfolder_data_proportion)[0]
        subsubfolder_id = random.choices(range(len(self.traj_dirs[subfolder_id])), weights=self.subsubfolder_data_proportion[subfolder_id])[0]
        datafolder_dirs = self.traj_dirs[subfolder_id][subsubfolder_id]

        idx_start = self.history_length + 1
        idx_end = len(datafolder_dirs) - self.pred_length - 1
        idx = random.randint(idx_start, idx_end)

        TIME_BUFF = []
        # print("start idx: {0}".format(str(t0 - history_length)))
        for i in range(idx - self.history_length, idx):
            vehicle_list = pickle.load(open(datafolder_dirs[i], "rb"))
            TIME_BUFF.append(vehicle_list)

        return TIME_BUFF

    def _cal_travel_distance(self, TIME_BUFF_last_moment, TIME_BUFF_newly_generated):
        TIME_BUFF = TIME_BUFF_last_moment + TIME_BUFF_newly_generated
        traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)

        travel_distance = []
        for vid, v_info in traj_pool.pool.items():
            if len(v_info['vehicle']) == 1:  # This vehicle just been generated
                continue
            v_loc = []
            for v_item in v_info['vehicle']:
                wd_x, wd_y = v_item.location.x, v_item.location.y
                v_loc.append([wd_x, wd_y])

            v_loc_array = np.array(v_loc)  # num_of_steps x 2
            v_loc_diff = np.diff(v_loc_array, axis=0)
            travel_distance_each_step = np.linalg.norm(v_loc_diff, axis=1)
            dist = np.sum(travel_distance_each_step)
            travel_distance.append(dist)

        total_travel_distance = sum(travel_distance)
        return total_travel_distance

    def generate_simulation_metric(self, one_sim_TIME_BUFF):
        """
        The input is the simulation episode index and the simulated trajectories of this episode.
        This function will calculate the simulation metrics (e.g., vehicle speed, distance, etc.)
        """
        if self.interpolate_flag:
            # interpolate the trajectory to a finer resolution first for analysis
            evaluate_metric_TIME_BUFF, new_resol = self.traj_interpolator.interpolate_traj(one_sim_TIME_BUFF, intep_steps=self.intep_steps, original_resolution=self.sim_resol)
            self.SimMetricsAnalyzer.sim_resol = new_resol
        else:
            evaluate_metric_TIME_BUFF = one_sim_TIME_BUFF
        self._gen_sim_metric(one_sim_TIME_BUFF=evaluate_metric_TIME_BUFF)

    def update_basic_sim_metric(self, sim_idx):
        if self.one_sim_colli_flag:
            self.colli_num += 1
        self.total_sim_steps += self.one_sim_steps_all_vehicles
        self.total_travel_distances += self.one_sim_travel_distances_all_vehicles
        self.total_sim_wall_time += self.one_sim_wall_time

        # if sim_idx % 100 == 1:
        if True:  # when sim_wall_time is very long (e.g., 30 mins)
            print(f'Sim number:{sim_idx + 1}, colli num: {self.colli_num}, total sim wall time: '
                  f'{self.total_sim_wall_time} (s), total travel dist: {self.total_travel_distances} (m) \n')
            res_df = pd.DataFrame(np.array([sim_idx + 1, self.colli_num, self.total_sim_wall_time, self.total_travel_distances]).reshape(1, -1),
                                  columns=['Sim number', 'colli num', 'total sim wall time', 'total travel dist'])

            res_df_save_path = os.path.join(self.realistic_metric_save_folder, "res_df.csv")
            res_df.to_csv(res_df_save_path, index=False)

    def run_simulations(self, sim_num, initial_TIME_BUFF=None):
        """
        Run simulations
        """

        for sim_idx in range(sim_num):

            # Run one episode of simulation
            self.run_a_sim(sim_idx=sim_idx, initial_TIME_BUFF=initial_TIME_BUFF)

            # Generate and save simulation metrics
            if self.gen_realistic_metric_flag:
                self.generate_simulation_metric(self.one_sim_TIME_BUFF)
                self.save_sim_metric()
                print("Saving sim metrics!")

            # Save simulation video
            if self.save_viz_flag:
                self.save_time_buff_video(self.one_sim_TIME_BUFF, background_map=self.background_map, file_name=sim_idx, save_path=self.save_viz_folder)

            # Save simulated trajectories
            if self.save_simulated_TIME_BUFF_flag:
                save_viz_path = os.path.join(self.simulated_TIME_BUFF_save_folder, str(sim_idx))
                self.save_trajectory(self.one_sim_TIME_BUFF, save_path=save_viz_path, sim_id=sim_idx, step_id=0)

            # Update, print, and save basic simulation results
            self.update_basic_sim_metric(sim_idx)

    def _gen_sim_metric(self, one_sim_TIME_BUFF):
        # Construct traj dataframe
        self.SimMetricsAnalyzer.construct_traj_data(one_sim_TIME_BUFF)

        # PET analysis
        if self.gen_realistic_metric_dict["PET"]:
            PET_list = self.SimMetricsAnalyzer.PET_analysis()
            self.output_PET_list.append(PET_list)

        # In circle instant speed analysis
        if self.gen_realistic_metric_dict["instant_speed"]:
            instant_speed_list = self.SimMetricsAnalyzer.in_circle_instant_speed_analysis()
            self.output_instant_speed_list.append(instant_speed_list)

        # yielding distance and speed analysis
        if self.gen_realistic_metric_dict["yielding_speed_and_distance"]:
            yielding_conflict_dist_and_v_dict = self.SimMetricsAnalyzer.yilding_distance_and_speed_analysis()
            self.output_yielding_conflict_dist_and_v_dict_list.append(yielding_conflict_dist_and_v_dict)

        # all positions distance distribution analysis
        if self.gen_realistic_metric_dict["distance"]:
            distance_all_vehicle_pairs_list_three_circle = self.SimMetricsAnalyzer.distance_analysis(mode='three_circle', only_in_roundabout_circle=False)
            self.output_distance_all_vehicle_pairs_list_three_circle.append(distance_all_vehicle_pairs_list_three_circle)

    def save_sim_metric(self):
        if self.gen_realistic_metric_dict["PET"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_PET_list.json"), 'w') as f:
                json.dump(self.output_PET_list, f, indent=4)

        if self.gen_realistic_metric_dict["instant_speed"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_instant_speed_list.json"), 'w') as f:
                json.dump(self.output_instant_speed_list, f, indent=4)

        if self.gen_realistic_metric_dict["yielding_speed_and_distance"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_yielding_conflict_dist_and_v_dict_list.json"), 'w') as f:
                json.dump(self.output_yielding_conflict_dist_and_v_dict_list, f, indent=4)

        if self.gen_realistic_metric_dict["distance"]:
            with open(os.path.join(self.realistic_metric_save_folder, "output_distance_list_three_circle.json"), 'w') as f:
                json.dump(self.output_distance_all_vehicle_pairs_list_three_circle, f, indent=4)

    def initialize_sim(self, initial_TIME_BUFF=None):
        """
        Initialize a simulation episode by sampling from trajectory clips or given TIME_BUFF.
        """
        # read frames and initialize trajectory pool
        initial_has_collision = True
        while initial_has_collision:
            TIME_BUFF = self._initialize_sim() if initial_TIME_BUFF is None else initial_TIME_BUFF
            TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)  # remove initial vehicles in exit area.
            traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)
            initial_has_collision = self.sim.collision_check(TIME_BUFF)
            if initial_has_collision and initial_TIME_BUFF is not None:
                raise ValueError("The given initial TIME BUFF is not safe, collision in it!")

        # Initialize some basic stats for this simulation episode
        self.one_sim_TIME_BUFF = TIME_BUFF
        self.one_sim_steps_all_vehicles = 0  # sum of number of all steps of all vehicles
        self.one_sim_travel_distances_all_vehicles = 0  # sum of all vehicles travel distances
        self.one_sim_wall_time = 0.  # [s] simulation wall time of this simulation episode
        self.one_sim_colli_flag = False
        self.one_sim_TIME_BUFF_last_moment = None
        self.one_sim_TIME_BUFF_newly_generated = None
        self.one_sim_whole_sim_do_safety_mapping_flag = True

        return TIME_BUFF, traj_pool

    def visualize_time_buff(self, TIME_BUFF, tt=0):
        # visualization and write result video
        if self.viz_flag:
            if self.interpolate_flag:
                visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF, intep_steps=self.intep_steps, original_resolution=self.sim_resol)
                freq = self.intep_steps + 1
            else:
                visualize_TIME_BUFF = TIME_BUFF
                freq = 1

            if tt == 0:
                self._visualize_time_buff(visualize_TIME_BUFF, self.background_map)
            else:
                self._visualize_time_buff(visualize_TIME_BUFF[-self.rolling_step * freq:], self.background_map)

    def run_one_sim_step(self, traj_pool, TIME_BUFF):
        """
        Run one simulation step
        """

        # run self-simulation
        pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid, current_lat, current_lon = self.sim.run_forwardpass(traj_pool)
        output_delta_position_mask = np.zeros(buff_vid.shape, dtype=bool)

        # determine whether to do safety mapping
        if self.use_neural_safety_mapping:
            if self.use_conflict_critic_module:
                tmp_TIME_BUFF = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)
                do_safety_mapping_flag = self.conflict_critic_agent.main_func(TIME_BUFF=tmp_TIME_BUFF)
            else:
                do_safety_mapping_flag = True  # Always use safety mapping
            self.one_sim_whole_sim_do_safety_mapping_flag = self.one_sim_whole_sim_do_safety_mapping_flag and do_safety_mapping_flag

            if do_safety_mapping_flag:
                pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, output_delta_position_mask = self.sim.do_safety_mapping(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, buff_vid,
                                                                                                                                          output_delta_position_mask=True)
            else:
                print('Generate a collision!')

        TIME_BUFF = self.sim.prediction_to_trajectory_rolling_horizon(pred_lat, pred_lon, pred_cos_heading, pred_sin_heading, pred_vid, TIME_BUFF, rolling_step=self.rolling_step)

        return TIME_BUFF, pred_vid, output_delta_position_mask

    def update_basic_stats_of_the_current_sim_episode(self, tt, TIME_BUFF, pred_vid):
        if tt == 0:
            self.one_sim_TIME_BUFF_newly_generated = TIME_BUFF
            self.one_sim_TIME_BUFF_last_moment = TIME_BUFF[:1]
        else:
            self.one_sim_TIME_BUFF_newly_generated = TIME_BUFF[-self.rolling_step:]

        # Total steps
        num_vehicle_in_the_sim = (pred_vid[~np.isnan(pred_vid)]).shape[0] / self.pred_length
        self.one_sim_steps_all_vehicles += (num_vehicle_in_the_sim * self.rolling_step)
        # Travel distances
        travel_distance = self._cal_travel_distance(self.one_sim_TIME_BUFF_last_moment, self.one_sim_TIME_BUFF_newly_generated)
        self.one_sim_travel_distances_all_vehicles += travel_distance
        self.one_sim_TIME_BUFF_last_moment = TIME_BUFF[-1:]

    def run_a_sim(self, sim_idx, initial_TIME_BUFF=None):
        """
        Run one simulation
        """

        # Initialize the simulation
        TIME_BUFF, traj_pool = self.initialize_sim(initial_TIME_BUFF=initial_TIME_BUFF)

        # Run this simulation episode
        for tt in range(self.max_m_steps):

            # Visualize frames
            self.visualize_time_buff(TIME_BUFF, tt=tt)

            # Run one simulation step
            TIME_BUFF, pred_vid, output_delta_position_mask = self.run_one_sim_step(traj_pool=traj_pool, TIME_BUFF=TIME_BUFF)

            # Remove out of bound vehicles
            TIME_BUFF = self.sim.remove_out_of_bound_vehicles(TIME_BUFF, dataset=self.dataset)

            # Generate entering vehicles at source points.
            TIME_BUFF = self.traffic_generator.generate_veh_at_source_pos(TIME_BUFF)  # Generate entering vehicles at source points.
            traj_pool = self.sim.time_buff_to_traj_pool(TIME_BUFF)
            self.one_sim_TIME_BUFF += TIME_BUFF[-self.rolling_step:]

            # Update basic stats of this current simulation episode.
            self.update_basic_stats_of_the_current_sim_episode(tt, TIME_BUFF, pred_vid)

            # Collision check. If collision, save out crash video and trajectory data if the flag is True.
            self.one_sim_colli_flag = self.sim.collision_check(self.one_sim_TIME_BUFF_newly_generated)
            if self.one_sim_colli_flag:
                # Visualize frames
                self.visualize_time_buff(TIME_BUFF, tt=tt)

                if self.save_collision_data_flag:
                    save_steps = 2
                    save_TIME_BUFF = self.one_sim_TIME_BUFF[-save_steps:]
                    collision_data_path = os.path.join(self.realistic_metric_save_folder, 'crash_frame')
                    self.save_trajectory(save_TIME_BUFF, save_path=collision_data_path, sim_id=sim_idx, step_id=0)

                break

        self.one_sim_wall_time = (tt + 1) * self.sim_resol * self.rolling_step

    def _visualize_time_buff(self, TIME_BUFF, background_map):
        for i in range(len(TIME_BUFF)):
            vehicle_list = TIME_BUFF[i]
            vis = background_map.render(vehicle_list, with_traj=True, linewidth=6)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
            cv2.imshow('vis', img)  # rgb-->bgr
            cv2.waitKey(1)

    def save_time_buff_video(self, TIME_BUFF, background_map, file_name, save_path, color_vid_list=None, with_traj=True):
        if self.interpolate_flag:
            visualize_TIME_BUFF, _ = self.traj_interpolator.interpolate_traj(TIME_BUFF,
                                                                             intep_steps=self.intep_steps,
                                                                             original_resolution=self.sim_resol)
        else:
            visualize_TIME_BUFF = TIME_BUFF

        os.makedirs(save_path, exist_ok=True)
        collision_video_writer = cv2.VideoWriter(save_path + r'/{0}.mp4'.format(file_name), cv2.VideoWriter_fourcc(*'MP4V'), self.save_fps, (background_map.w, background_map.h))
        for i in range(len(visualize_TIME_BUFF)):
            vehicle_list = visualize_TIME_BUFF[i]
            vis = background_map.render(vehicle_list, with_traj=with_traj, linewidth=6, color_vid_list=color_vid_list)
            img = vis[:, :, ::-1]
            # img = cv2.resize(img, (768, int(768 * background_map.h / background_map.w)))  # resize when needed
            collision_video_writer.write(img)

    def save_trajectory(self, TIME_BUFF, save_path, sim_id, step_id):
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(TIME_BUFF)):
            vehicle_list = TIME_BUFF[i]
            frame_id = step_id + i
            output_file_path = os.path.join(save_path, str(sim_id) + '-' + str(frame_id).zfill(5) + '.pickle')
            pickle.dump(vehicle_list, open(output_file_path, "wb"))
