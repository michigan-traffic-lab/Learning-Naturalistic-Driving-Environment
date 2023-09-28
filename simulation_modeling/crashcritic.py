import random
import numpy as np
import pandas as pd
import pickle
from itertools import combinations
from shapely.geometry import Polygon, Point, LineString
import copy

from vehicle import Vehicle, Size3d
from trajectory_pool import TrajectoryPool


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


class CrashCritic(object):

    def __init__(self, sim_resol=0.4):
        self.traj_pool = None
        self.traj_df = None
        self.sim_resol = sim_resol  # simulation resolution in [s]
        # Acceptance probability of different crash type. Calibrated for AA_rdbt.
        # There is no data to calibrate for the rounD dataset, so we use the same probability as AA_rdbt.
        self.type_prob = {'rear_end': 4.775061124694377e-05, 'angle': 0.0028623333333333335,
                          'sideswipe': 0.0008131147540983608, 'other': 0.0, 'more_than_two_vehicles': 0.0}

    @staticmethod
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

    def time_buff_to_traj_pool(self, TIME_BUFF):
        traj_pool = TrajectoryPool(max_missing_age=float("inf"), road_matcher=None, ROI_matcher=None)
        for i in range(len(TIME_BUFF)):
            traj_pool.update(TIME_BUFF[i], ROI_matching=False)
        return traj_pool

    def dynamics_check(self, TIME_BUFF, collision_vehicle_list, history_length=5, v_ub=15, a_ub=10, a_lb=-10):

        # Construct traj pool
        self.traj_pool = self.time_buff_to_traj_pool(TIME_BUFF[-history_length-2:])
        self.traj_df = pd.DataFrame(columns=['vid', 'x', 'y', 'heading', 'region_position', 'yielding_area', 'at_circle_lane', 't', 'update', 'vehicle', 'dt', 'missing_days'])

        # Construct traj df
        for v in collision_vehicle_list:
            vid = v.id
            self.traj_df = self.traj_df.append(pd.DataFrame.from_dict(self.traj_pool.pool[vid]), ignore_index=True)

        dynamics_satisfy_flag = True
        for v in collision_vehicle_list:
            vid = v.id
            v_traj = self.traj_df[self.traj_df['vid'] == vid]

            if v_traj.shape[0] <= 4:  # The vehicle just show up.
                dynamics_satisfy_flag = False
                break

            v_traj.loc[:, "dt"] = v_traj["t"].diff()
            v_traj.loc[:, "dx"] = v_traj["x"].diff()
            v_traj.loc[:, "dy"] = v_traj["y"].diff()
            v_traj.loc[:, "travel_distance"] = (v_traj["dx"] ** 2 + v_traj["dy"] ** 2) ** 0.5

            try:
                assert (v_traj.dt.dropna() > 0).all()
            except:
                dynamics_satisfy_flag = False
                break

            v_traj.loc[:, "instant_speed"] = v_traj['travel_distance'] / (v_traj['dt'] * self.sim_resol)  # [m/s]
            instant_speed_tmp = v_traj['instant_speed'].dropna().tolist()
            if (np.array(instant_speed_tmp) > v_ub).any():
                dynamics_satisfy_flag = False
                break

            v_traj.loc[:, "speed_change"] = v_traj["instant_speed"].diff()
            v_traj.loc[:, "acceleration"] = v_traj['speed_change'] / (v_traj['dt'] * self.sim_resol)  # [m/s]
            acceleration_tmp = v_traj['acceleration'].dropna().tolist()
            if (np.array(acceleration_tmp) > a_ub).any() or (np.array(acceleration_tmp) < a_lb).any():
                dynamics_satisfy_flag = False
                break

        return dynamics_satisfy_flag

    @staticmethod
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

    def main_func(self, TIME_BUFF):
        """
        Determine whether accept a generated collision (not passing safety mapping network).
        Two requirements: 1. has collision, 2. satisfy acceleration/velocity bounds.
        Then has different probability for accepting different collision types.
        """
        do_safety_mapping_flag = True
        collision_flag, collision_vehicle_list = self.collision_check(vehicle_list=TIME_BUFF[-1])
        if not collision_flag:
            return do_safety_mapping_flag

        dynamics_satisfy_flag = self.dynamics_check(TIME_BUFF, collision_vehicle_list, history_length=5, v_ub=10, a_ub=10, a_lb=-10)
        if not dynamics_satisfy_flag:
            return do_safety_mapping_flag

        collision_type, _, _ = self.collision_type_classification(collision_vehicle_list)
        prob = self.type_prob[collision_type]  # Probability of accepting this collision

        sample_flag = np.random.uniform() < prob
        if sample_flag:
            do_safety_mapping_flag = False

        return do_safety_mapping_flag
