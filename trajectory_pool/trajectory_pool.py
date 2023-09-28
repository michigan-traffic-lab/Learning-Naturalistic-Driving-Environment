import numpy as np
import copy
import matplotlib.pyplot as plt


class TrajectoryPool(object):
    """
    A tool for managing trajectories (longitudinal data).
    """

    def __init__(self, max_missing_age=5, road_matcher=None, ROI_matcher=None):
        self.pool = {}  # vehicle pool
        self.max_missing_age = max_missing_age  # vehicles with > max_missing_age will be removed
        self.t_latest = 0  # time stamp of latest frame

        self.road_matcher = road_matcher  # drivable map matcher, used to get pixel points and map to drivable map.
        self.ROI_matcher = ROI_matcher  # used to match a position to ROI.

    def update(self, vehicle_list, ROI_matching=False):

        self.t_latest += 1

        # all vehicles missing days +1
        for vid, value in self.pool.items():
            self.pool[vid]['missing_days'] += 1

        # update trajectory pool
        for i in range(len(vehicle_list)):
            v = vehicle_list[i]
            if v.location.x == None or v.location.y == None:
                continue

            if ROI_matching:
                pxl_pt = self.road_matcher._world2pxl([v.location.x, v.location.y])
                pxl_pt[1] = np.clip(pxl_pt[1], a_min=0, a_max=self.road_matcher.road_map.shape[0]-1)
                pxl_pt[0] = np.clip(pxl_pt[0], a_min=0, a_max=self.road_matcher.road_map.shape[1]-1)
                v.region_position = self.ROI_matcher.region_position_matching(pxl_pt)
                v.yielding_area = self.ROI_matcher.yielding_area_matching(pxl_pt)
                v.at_circle_lane = self.ROI_matcher.at_circle_lane_matching(pxl_pt)

            if v.id not in self.pool.keys():
                self.pool[v.id] = {'vid': [str(v.id)], 'update': False, 'vehicle': [v], 'dt': [1], 't': [self.t_latest], 'missing_days': 0,
                                   'x': [v.location.x], 'y': [v.location.y], 'heading': [v.speed_heading],
                                   'region_position': [v.region_position if ROI_matching else None],
                                   'yielding_area': [v.yielding_area if ROI_matching else None], 'at_circle_lane': [v.at_circle_lane if ROI_matching else None]}
            else:
                self.pool[v.id]['vehicle'].append(v)
                self.pool[v.id]['update'] = True  # means this vehicle's just updated
                self.pool[v.id]['dt'].append(copy.deepcopy(self.pool[v.id]['missing_days']))  # dt since last time saw it
                self.pool[v.id]['t'].append(self.t_latest)  # time stamp
                self.pool[v.id]['missing_days'] = 0  # just saw it, so clear missing days
                self.pool[v.id]['vid'].append(str(v.id))
                self.pool[v.id]['x'].append(v.location.x)
                self.pool[v.id]['y'].append(v.location.y)
                self.pool[v.id]['heading'].append(v.speed_heading)
                self.pool[v.id]['region_position'].append(v.region_position if ROI_matching else None)
                self.pool[v.id]['yielding_area'].append(v.yielding_area if ROI_matching else None)
                self.pool[v.id]['at_circle_lane'].append(v.at_circle_lane if ROI_matching else None)

        # remove dead traj id (missing for a long time)
        for vid, value in self.pool.copy().items():
            if self.pool[vid]['missing_days'] > self.max_missing_age:
                del self.pool[vid]

    def vehicle_id_list(self):
        return list(self.pool.keys())

    def flatten_trajectory(self, max_num_vehicles, time_length, output_vid=False):

        # create lat and lon and heading buffer
        veh_num = len(list(self.pool.keys()))
        buff_lat = np.empty([veh_num, time_length])
        buff_lat[:] = np.nan
        buff_lon = np.empty([veh_num, time_length])
        buff_lon[:] = np.nan
        buff_cos_heading = np.empty([veh_num, time_length])
        buff_cos_heading[:] = np.nan
        buff_sin_heading = np.empty([veh_num, time_length])
        buff_sin_heading[:] = np.nan
        buff_vid = np.empty([veh_num, time_length])
        buff_vid[:] = np.nan

        # fill-in lon and lat and heading buffer
        i = 0
        for _, traj in self.pool.items():
            ts = traj['t']
            vs = traj['vehicle']
            for j in range(len(ts)):
                lat, lon = vs[j].location.x, vs[j].location.y
                heading = np.radians(vs[j].speed_heading)  # Convert degrees to radians
                if lat is None:
                    continue
                t = self.t_latest - ts[j]
                if t >= time_length:
                    continue
                buff_lat[i, t] = lat
                buff_lon[i, t] = lon
                buff_cos_heading[i, t] = np.cos(heading)
                buff_sin_heading[i, t] = np.sin(heading)
            i += 1

        # fill-in id buffer
        i = 0
        for _, traj in self.pool.items():
            vs = traj['vehicle']
            buff_vid[i, :] = vs[-1].id
            i += 1

        buff_lat = buff_lat[:, ::-1]
        buff_lon = buff_lon[:, ::-1]
        buff_cos_heading = buff_cos_heading[:, ::-1]
        buff_sin_heading = buff_sin_heading[:, ::-1]

        # pad or crop to m x max_num_vehicles
        buff_lat = self._fixed_num_vehicles(buff_lat, max_num_vehicles)
        buff_lon = self._fixed_num_vehicles(buff_lon, max_num_vehicles)
        buff_cos_heading = self._fixed_num_vehicles(buff_cos_heading, max_num_vehicles)
        buff_sin_heading = self._fixed_num_vehicles(buff_sin_heading, max_num_vehicles)
        buff_vid = self._fixed_num_vehicles(buff_vid, max_num_vehicles)

        if output_vid:
            return buff_lat, buff_lon, buff_cos_heading, buff_sin_heading, buff_vid
        else:
            return buff_lat, buff_lon, buff_cos_heading, buff_sin_heading

    @staticmethod
    def _fixed_num_vehicles(x, max_num_vehicles):

        m_veh, l = x.shape
        if m_veh >= max_num_vehicles:
            x_ = x[0:max_num_vehicles, :]
        else:
            x_ = np.empty([max_num_vehicles, l], dtype=x.dtype)
            x_[:] = np.nan
            x_[0:m_veh, :] = x

        return x_

