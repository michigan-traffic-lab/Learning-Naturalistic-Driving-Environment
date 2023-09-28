import numpy as np
import os
import pickle

from vehicle import Vehicle, Size3d
from road_matching import RoadMatcher, ROIMatcher


class TrafficGenerator(object):
    """
        Generate vehicles during long-time traffic simulation (inference time).
    """

    def __init__(self, config):

        self.method = config["method"]
        self.check_safety_method = config['check_safety_method']
        try:
            assert (self.check_safety_method == 'uniform_safety_check' or self.check_safety_method == 'lane_based_safety_check')
        except:
            raise NotImplementedError(
                'Wrong safety check method for initializing vehicles %s (choose one from [uniform_safety_check, lane_based_safety_check])' % self.check_safety_method)
        if self.check_safety_method == 'uniform_safety_check':
            # Use Euclidean distance with all other vehicles regardless lane information. If you just want to quickly run the simulation, use this method.
            self.uniform_safety_buffer = config["uniform_safety_buffer"]
            print('Using uniform safety check for initializing vehicles!')
        if self.check_safety_method == 'lane_based_safety_check':  # the safety buffer of Euclidean distance for vehicle within in the lane is different from those not in the same lane.
            self.same_lane_safety_buffer = config["same_lane_safety_buffer"]
            self.different_lane_safety_buffer = config["different_lane_safety_buffer"]
            self.road_matcher = RoadMatcher(map_file_dir=config["drivable_map_dir"], map_height=config["map_height"], map_width=config["map_width"])
            self.ROI_matcher = ROIMatcher(entrance_map_dir=config["entrance_map_dir"], map_height=config["map_height"], map_width=config["map_width"])
            print('Using lane-based safety check for initializing vehicles!')
        # self.safety_buffer = config["safety_buffer"]

        if self.method == "Poisson":
            self.sim_ros = config["sim_resol"]
            self.default_poisson_arr_rate = config["default_Poisson_arr_rate"]  # veh/hr
            self.default_dt_arr_rate = self.default_poisson_arr_rate / (3600 / self.sim_ros)  # veh/each resolution
        elif self.method == "Random":
            self.default_random_gen_prob = config["default_random_gen_prob"]

        # Info defined in each map class.
        self.LAT0, self.LON0 = None, None
        self.global_source_pos = []  # source positions in lat lon. Defined in each map class.
        self.local_source_pos = []  # source positions in local coord.
        self.poisson_arr_rate_each_pos, self.dt_arr_rate_each_pos = [], []  # Poisson arrival rate of each position. Defined in each map class.
        self.random_gen_prob_each_pos = []  # Random generation probability of each position. Defined in each map class.

        self.default_id = 99999999
        self.gen_num = 0

        self.v_modeling_size = Size3d(width=1.8, length=3.6, height=1.5)
        self.v_modeling_safe_size = Size3d(width=2.0, length=3.8, height=1.5)

        self.np_random = np.random.RandomState()  # Random seed to guarantee reproducible. Put a fixed number in it to achieve reproducibility.

    def _gen(self, arr_or_gen_prob=None):
        """
        Check whether generate a vehicle
        """
        if self.method == "Poisson":
            return self._possion_process_gen(arr_or_gen_prob)
        elif self.method == "Random":
            return self._random_gen(arr_or_gen_prob)
        else:
            raise NotImplementedError(
                '%s does not supported yet...Choose from ["Poisson", "Random"].' % self.method)

    def _possion_process_gen(self, dt_arr_rate=None):
        dt_arr_rate = dt_arr_rate if dt_arr_rate is not None else self.default_dt_arr_rate

        p0 = np.exp(-dt_arr_rate)  # Poisson prob P(0)
        p_gen = 1. - p0
        # rand_number = random.uniform(0, 1)
        rand_number = self.np_random.uniform(0, 1)
        if rand_number <= p_gen:
            return True
        else:
            return False

    def _random_gen(self, random_gen_prob=None):
        """
        Generate vehicle by given probability.
        """
        random_gen_prob = random_gen_prob if random_gen_prob is not None else self.default_random_gen_prob

        # rand_number = random.uniform(0, 1)
        rand_number = self.np_random.uniform(0, 1)
        if rand_number <= random_gen_prob:
            return True
        else:
            return False

    def generate_veh_at_source_pos(self, TIME_BUFF):
        """
        Generate new vehicles at source positions
        """
        CURRENT_VEH_LIST = TIME_BUFF[-1]

        # Loop over all sources
        num_sources = len(self.global_source_pos)
        # loop_order = list(np.random.choice(list(range(num_sources)), size=num_sources, replace=False))
        loop_order = list(self.np_random.choice(list(range(num_sources)), size=num_sources, replace=False))
        for idx in loop_order:
            source = self.global_source_pos[idx]
            source_name = self.global_source_name[idx]
            arr_or_gen_prob = None
            if self.method == 'Poisson':
                arr_or_gen_prob = self.dt_arr_rate_each_pos[idx] if len(self.dt_arr_rate_each_pos) > 0 else self.default_dt_arr_rate
            if self.method == 'Random':
                arr_or_gen_prob = self.random_gen_prob_each_pos[idx] if len(self.random_gen_prob_each_pos) > 0 else self.default_random_gen_prob
            gen_flag = self._gen(arr_or_gen_prob)

            if gen_flag:
                safe_flag = None
                # v = random.sample(source, 1)[0]
                v = self.np_random.choice(source, 1)[0]
                v.id = str(self.default_id + self.gen_num)

                if self.check_safety_method == 'uniform_safety_check':
                    safe_flag = self._uniform_check_safety(CURRENT_VEH_LIST, gen_lat=v.location.x, gen_lon=v.location.y)
                if self.check_safety_method == 'lane_based_safety_check':
                    safe_flag = self._lane_based_check_safety(CURRENT_VEH_LIST, gen_lat=v.location.x, gen_lon=v.location.y, entrance_name=source_name)

                if safe_flag is True:
                    self.gen_num += 1
                    CURRENT_VEH_LIST.append(v)

        TIME_BUFF[-1] = CURRENT_VEH_LIST

        return TIME_BUFF

    def _lane_based_check_safety(self, CURRENT_VEH_LIST, gen_lat, gen_lon, entrance_name):
        """
        Check whether the initial pos is safe.
        This method will check the safety based on existing vehicle lane. When the existing vehicle is in the same lane of the newly generated vehicle,
        the safety threshold is larger since it concerns with the vehicle length. When the existing vehicle is not in the same lane, the threshold is
        smaller since it concerns with the vehicle width.
        This method need the entrance map information.
        """
        if len(CURRENT_VEH_LIST) == 0:
            return True
        same_lane_v_list = []
        different_lane_v_list = []

        for v in CURRENT_VEH_LIST:
            pxl_pt = self.road_matcher._world2pxl([v.location.x, v.location.y])
            pxl_pt[1] = np.clip(pxl_pt[1], a_min=0, a_max=self.road_matcher.road_map.shape[0] - 1)
            pxl_pt[0] = np.clip(pxl_pt[0], a_min=0, a_max=self.road_matcher.road_map.shape[1] - 1)
            region_name = self.ROI_matcher.entrance_lane_matching(pxl_pt)
            if region_name == entrance_name:
                same_lane_v_list.append(v)
            else:
                different_lane_v_list.append(v)

        same_lane_veh_local_pos = np.array([(v.location.x, v.location.y) for v in same_lane_v_list])
        different_lane_veh_local_pos = np.array([(v.location.x, v.location.y) for v in different_lane_v_list])
        gen_local_pos = np.array([(gen_lat, gen_lon)])
        if same_lane_veh_local_pos.shape[0] != 0:   # No vehicles at the same entrance now.
            dis_with_in_the_same_lane_veh = np.linalg.norm(same_lane_veh_local_pos - gen_local_pos, axis=1)
            safe_with_the_same_lane = ((dis_with_in_the_same_lane_veh >= self.same_lane_safety_buffer).all()).item()
        else:
            safe_with_the_same_lane = True

        if different_lane_veh_local_pos.shape[0] != 0:
            dis_with_different_lane_veh = np.linalg.norm(different_lane_veh_local_pos - gen_local_pos, axis=1)
            safe_with_different_lane = ((dis_with_different_lane_veh >= self.different_lane_safety_buffer).all()).item()
        else:
            safe_with_different_lane = True

        safety_flag = safe_with_the_same_lane and safe_with_different_lane

        return safety_flag

    def _uniform_check_safety(self, CURRENT_VEH_LIST, gen_lat, gen_lon):
        """
        Check whether the initial pos is safe.
        This method is to check the Euclidean distance with all existing vehicles regardless which lane they are in.
        This method do not need the entrance map information.
        If you just want to quickly run the simulation and not sure whether to use _lane_based_check_safety, use this method.
        """
        if len(CURRENT_VEH_LIST) == 0:
            return True
        all_veh_local_pos = np.array([(v.location.x, v.location.y) for v in CURRENT_VEH_LIST])
        gen_local_pos = np.array([(gen_lat, gen_lon)])
        dis_with_all_veh = np.linalg.norm(all_veh_local_pos - gen_local_pos, axis=1)
        safety_flag = not ((dis_with_all_veh < self.uniform_safety_buffer).any()).item()

        return safety_flag


class AA_rdbt_TrafficGenerator(TrafficGenerator):

    def __init__(self, config):
        super(AA_rdbt_TrafficGenerator, self).__init__(config)

        initial_data_path = config["gen_veh_states_dir"]
        initial_res_dict = pickle.load(open(os.path.join(initial_data_path, 'initial_vehicle_dict.pickle'), "rb"))

        self.n_in1, self.n_in2 = initial_res_dict['n_in1'], initial_res_dict['n_in2']
        self.e_in1, self.e_in2 = initial_res_dict['e_in1'], initial_res_dict['e_in2']
        self.s_in1, self.s_in2 = initial_res_dict['s_in1'], initial_res_dict['s_in2']
        self.w_in1, self.w_in2 = initial_res_dict['w_in1'], initial_res_dict['w_in2']

        self.n_in3 = initial_res_dict['n_in3']  # dedicated right-turn lane

        self.global_source_pos = [self.n_in1, self.n_in2,
                                  self.e_in1, self.e_in2,
                                  self.s_in1, self.s_in2,
                                  self.w_in1, self.w_in2,
                                  self.n_in3]
        self.global_source_name = ['entrance_n_1', 'entrance_n_2',
                                   'entrance_e_1', 'entrance_e_2',
                                   'entrance_s_1', 'entrance_s_2',
                                   'entrance_w_1', 'entrance_w_2',
                                   'entrance_n_rightturn']  # Make sure the name here is consistent with that in ROI_matcher.entrance_lane_matching.

        self.local_source_pos = [(loc[0], loc[1]) for loc in self.global_source_pos]

        self.poisson_arr_rate_each_pos = [402.0, 208.0, 424.0, 460.0, 258.0, 250.0, 458.0, 150.0, 236.0]  # veh/h based on AA roundabout calibrated
        self.dt_arr_rate_each_pos = [poisson_arr_rate / (3600 / self.sim_ros) for poisson_arr_rate in self.poisson_arr_rate_each_pos]  # veh/each resolution


class rounD_TrafficGenerator(TrafficGenerator):

    def __init__(self, config):
        super(rounD_TrafficGenerator, self).__init__(config)

        initial_data_path = config["gen_veh_states_dir"]
        initial_res_dict = pickle.load(open(os.path.join(initial_data_path, 'initial_vehicle_dict.pickle'), "rb"))

        self.n_in1, self.n_in2 = initial_res_dict['n_in1'], initial_res_dict['n_in2']
        self.e_in1, self.e_in2 = initial_res_dict['e_in1'], initial_res_dict['e_in2']
        self.s_in1, self.s_in2 = initial_res_dict['s_in1'], initial_res_dict['s_in2']
        self.w_in1, self.w_in2 = initial_res_dict['w_in1'], initial_res_dict['w_in2']

        self.n_in3 = initial_res_dict['n_in3']  # dedicated right-turn lane
        self.s_in3 = initial_res_dict['s_in3']  # dedicated right-turn lane

        self.global_source_pos = [self.n_in1, self.n_in2,
                                  self.e_in1, self.e_in2,
                                  self.s_in1, self.s_in2,
                                  self.w_in1, self.w_in2,
                                  self.n_in3, self.s_in3]
        self.global_source_name = ['entrance_n_1', 'entrance_n_2',
                                   'entrance_e_1', 'entrance_e_2',
                                   'entrance_s_1', 'entrance_s_2',
                                   'entrance_w_1', 'entrance_w_2',
                                   'entrance_n_rightturn', 'entrance_s_rightturn']  # Make sure the name here is consistent with that in ROI_matcher.entrance_lane_matching.

        self.local_source_pos = [(loc[0], loc[1]) for loc in self.global_source_pos]

        self.poisson_arr_rate_each_pos = [35, 494, 15, 406, 24, 387, 18, 427, 52, 72]  # veh/h based on rounD
        self.dt_arr_rate_each_pos = [poisson_arr_rate / (3600 / self.sim_ros) for poisson_arr_rate in self.poisson_arr_rate_each_pos]  # veh/each resolution