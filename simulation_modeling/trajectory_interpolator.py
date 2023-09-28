import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from trajectory_pool import TrajectoryPool
from vehicle import Vehicle, Size3d


class TrajInterpolator(object):

    def __init__(self):
        pass

    @staticmethod
    def time_buff_to_traj_pool(TIME_BUFF):
        traj_pool = TrajectoryPool(max_missing_age=float("inf"), road_matcher=None, ROI_matcher=None)
        for i in range(len(TIME_BUFF)):
            traj_pool.update(TIME_BUFF[i], ROI_matching=False)
        return traj_pool

    @staticmethod
    def traj_pool_to_traj_df(traj_pool):

        traj_df = pd.DataFrame(columns=['vid', 'x', 'y', 'heading', 'region_position', 'yielding_area', 'at_circle_lane', 't', 'update', 'vehicle', 'dt', 'missing_days'])

        # Construct traj df
        for vid in traj_pool.vehicle_id_list():
            traj_df = traj_df.append(pd.DataFrame.from_dict(traj_pool.pool[vid]), ignore_index=True)

        return traj_df

    def _verify_heading_interpolation_results(self, start, end, heading_intep_res):
        """
        Plot figures to verify whether the heading interpolation results are reasonable.
        """
        plt.plot(np.linspace(0, 1.5, 100) * np.cos(np.radians(start)), np.linspace(0, 1.5, 100) * np.sin(np.radians(start)), label='start')
        plt.plot(np.linspace(0, 1.5, 100) * np.cos(np.radians(end)), np.linspace(0, 1.5, 100) * np.sin(np.radians(end)), label='end')

        for i in range(len(heading_intep_res)):
            v = heading_intep_res[i]
            plt.plot(np.linspace(0, 1.5, 100) * np.cos(np.radians(v)), np.linspace(0, 1.5, 100) * np.sin(np.radians(v)), label=str(i))

        plt.axis("equal")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.legend()
        plt.show()

    def interpolate_traj(self, TIME_BUFF, intep_steps, original_resolution):

        new_TIME_BUFF = []

        for i in range(len(TIME_BUFF)-1):
            intep_TIME_BUFF = [[] for _ in range(intep_steps)]  # newly interpolated time buff

            vehicle_list_current_step = TIME_BUFF[i]
            vehicle_list_next_step = TIME_BUFF[i+1]

            traj_pool_current_step = self.time_buff_to_traj_pool([vehicle_list_current_step])
            traj_pool_next_step = self.time_buff_to_traj_pool([vehicle_list_next_step])

            for v_idx in range(len(vehicle_list_current_step)):
                v_current_step = vehicle_list_current_step[v_idx]
                vid = v_current_step.id
                v_info_current_step = traj_pool_current_step.pool[vid]['vehicle'][0]
                if vid not in traj_pool_next_step.vehicle_id_list():  # the vehicle disappear at the next time step, then cannot intep it.
                    continue
                v_info_next_step = traj_pool_next_step.pool[vid]['vehicle'][0]

                x_intep_res = list(np.interp(x=list(range(1, 1+intep_steps)), xp=[0, 1 + intep_steps], fp=[v_info_current_step.location.x, v_info_next_step.location.x]))
                y_intep_res = list(np.interp(x=list(range(1, 1+intep_steps)), xp=[0, 1 + intep_steps], fp=[v_info_current_step.location.y, v_info_next_step.location.y]))
                # Heading here starting from east and anti-clockwise: east: 0 degree, north: +90 degree, west: +180...
                start, end = v_info_current_step.speed_heading % 360., v_info_next_step.speed_heading % 360.
                if end - start > 180.:
                    heading_intep_res = list(np.interp(x=list(range(1, 1 + intep_steps)), xp=[0, 1 + intep_steps], fp=[start, end - 360]))
                elif end - start < -180.:
                    heading_intep_res = list(np.interp(x=list(range(1, 1 + intep_steps)), xp=[0, 1 + intep_steps], fp=[start, end + 360]))
                else:
                    heading_intep_res = list(np.interp(x=list(range(1, 1 + intep_steps)), xp=[0, 1 + intep_steps], fp=[start, end]))
                heading_intep_res = [val % 360. for val in heading_intep_res]
                # self._verify_heading_interpolation_results(start, end, heading_intep_res)  # verify interpolation results.
                id_intep_res = [vid for _ in range(intep_steps)]
                size_intep_res = [v_current_step.size for _ in range(intep_steps)]
                safe_size_intep_res = [v_current_step.safe_size for _ in range(intep_steps)]

                for t_idx, (x_intep, y_intep, heading_intep, id_intep, size_intep, safe_size_intep) in \
                        enumerate(zip(x_intep_res, y_intep_res, heading_intep_res, id_intep_res, size_intep_res, safe_size_intep_res)):
                    v = Vehicle()
                    v.location.x, v.location.y = x_intep, y_intep
                    v.id = id_intep
                    v.speed_heading = heading_intep
                    v.size = size_intep
                    v.safe_size = safe_size_intep
                    v.update_poly_box_and_realworld_4_vertices()
                    v.update_safe_poly_box()
                    intep_TIME_BUFF[t_idx].append(v)

            intep_TIME_BUFF = [TIME_BUFF[i]] + intep_TIME_BUFF
            new_TIME_BUFF = new_TIME_BUFF + intep_TIME_BUFF

        new_TIME_BUFF = new_TIME_BUFF + [TIME_BUFF[-1]]  # add back the last moment data
        new_resolution = original_resolution / (intep_steps + 1)

        return new_TIME_BUFF, new_resolution
