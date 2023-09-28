import cv2
import os


class ROIMatcher(object):
    """
    This class includes different ROI maps, each point can be checked whether
    within the interested ROI.
    """

    def __init__(self, drivable_map_dir=None, sim_remove_vehicle_area_map_dir=None, circle_map_dir=None, entrance_map_dir=None,
                 exit_map_dir=None, crosswalk_map_dir=None, yielding_area_map_dir=None, at_circle_lane_map_dir=None,
                 map_height=1024, map_width=1024):
        self.drivable_map = None
        self.sim_remove_vehicle_area_map = None

        if drivable_map_dir is not None:
            self.drivable_map = cv2.imread(drivable_map_dir, cv2.IMREAD_GRAYSCALE)
            self.drivable_map = cv2.resize(self.drivable_map, (map_width, map_height))

        if sim_remove_vehicle_area_map_dir is not None:
            self.sim_remove_vehicle_area_map = cv2.imread(sim_remove_vehicle_area_map_dir, cv2.IMREAD_GRAYSCALE)
            self.sim_remove_vehicle_area_map = cv2.resize(self.sim_remove_vehicle_area_map, (map_width, map_height))

        if circle_map_dir is not None:
            self.circle_1_q_map = cv2.imread(os.path.join(circle_map_dir, 'circle_1_q-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.circle_2_q_map = cv2.imread(os.path.join(circle_map_dir, 'circle_2_q-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.circle_3_q_map = cv2.imread(os.path.join(circle_map_dir, 'circle_3_q-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.circle_4_q_map = cv2.imread(os.path.join(circle_map_dir, 'circle_4_q-map.jpg'), cv2.IMREAD_GRAYSCALE)

            self.circle_1_q_map = cv2.resize(self.circle_1_q_map, (map_width, map_height))
            self.circle_2_q_map = cv2.resize(self.circle_2_q_map, (map_width, map_height))
            self.circle_3_q_map = cv2.resize(self.circle_3_q_map, (map_width, map_height))
            self.circle_4_q_map = cv2.resize(self.circle_4_q_map, (map_width, map_height))

        if entrance_map_dir is not None:
            # N entrance
            self.entrance_n_1 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_n_1-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.entrance_n_2 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_n_2-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.entrance_n_rightturn = cv2.imread(os.path.join(entrance_map_dir, 'entrance_n_rightturn-map.jpg'), cv2.IMREAD_GRAYSCALE)

            self.entrance_n_1 = cv2.resize(self.entrance_n_1, (map_width, map_height))
            self.entrance_n_2 = cv2.resize(self.entrance_n_2, (map_width, map_height))
            self.entrance_n_rightturn = cv2.resize(self.entrance_n_rightturn, (map_width, map_height))

            # E entrance
            self.entrance_e_1 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_e_1-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.entrance_e_2 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_e_2-map.jpg'), cv2.IMREAD_GRAYSCALE)

            self.entrance_e_1 = cv2.resize(self.entrance_e_1, (map_width, map_height))
            self.entrance_e_2 = cv2.resize(self.entrance_e_2, (map_width, map_height))

            # S entrance
            self.entrance_s_1 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_s_1-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.entrance_s_2 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_s_2-map.jpg'), cv2.IMREAD_GRAYSCALE)
            s_rightturn_filepath = os.path.join(entrance_map_dir, 'entrance_s_rightturn-map.jpg')
            self.entrance_s_rightturn = cv2.imread(s_rightturn_filepath, cv2.IMREAD_GRAYSCALE) if os.path.isfile(s_rightturn_filepath) else None

            self.entrance_s_1 = cv2.resize(self.entrance_s_1, (map_width, map_height))
            self.entrance_s_2 = cv2.resize(self.entrance_s_2, (map_width, map_height))
            self.entrance_s_rightturn = cv2.resize(self.entrance_s_rightturn, (map_width, map_height)) if os.path.isfile(s_rightturn_filepath) else None

            # W entrance
            self.entrance_w_1 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_w_1-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.entrance_w_2 = cv2.imread(os.path.join(entrance_map_dir, 'entrance_w_2-map.jpg'), cv2.IMREAD_GRAYSCALE)

            self.entrance_w_1 = cv2.resize(self.entrance_w_1, (map_width, map_height))
            self.entrance_w_2 = cv2.resize(self.entrance_w_2, (map_width, map_height))

        if exit_map_dir is not None:
            # exit
            self.exit_n = cv2.imread(os.path.join(exit_map_dir, 'exit_n-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.exit_e = cv2.imread(os.path.join(exit_map_dir, 'exit_e-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.exit_s = cv2.imread(os.path.join(exit_map_dir, 'exit_s-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.exit_w = cv2.imread(os.path.join(exit_map_dir, 'exit_w-map.jpg'), cv2.IMREAD_GRAYSCALE)
            exit_n_rightturn_filepath = os.path.join(exit_map_dir, 'exit_n_rightturn-map.jpg')
            self.exit_n_rightturn = cv2.imread(exit_n_rightturn_filepath, cv2.IMREAD_GRAYSCALE) if os.path.isfile(exit_n_rightturn_filepath) else None
            exit_s_rightturn_filepath = os.path.join(exit_map_dir, 'exit_s_rightturn-map.jpg')
            self.exit_s_rightturn = cv2.imread(exit_s_rightturn_filepath, cv2.IMREAD_GRAYSCALE) if os.path.isfile(exit_s_rightturn_filepath) else None

            self.exit_n = cv2.resize(self.exit_n, (map_width, map_height))
            self.exit_n_rightturn = cv2.resize(self.exit_n_rightturn, (map_width, map_height)) if os.path.isfile(exit_n_rightturn_filepath) else None
            self.exit_e = cv2.resize(self.exit_e, (map_width, map_height))
            self.exit_s = cv2.resize(self.exit_s, (map_width, map_height))
            self.exit_s_rightturn = cv2.resize(self.exit_s_rightturn, (map_width, map_height)) if os.path.isfile(exit_s_rightturn_filepath) else None
            self.exit_w = cv2.resize(self.exit_w, (map_width, map_height))

        if crosswalk_map_dir is not None:
            self.crosswalk = cv2.imread(os.path.join(crosswalk_map_dir, 'crosswalk-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.crosswalk = cv2.resize(self.crosswalk, (map_width, map_height))

        if yielding_area_map_dir is not None:
            self.yielding_n = cv2.imread(os.path.join(yielding_area_map_dir, 'yielding_n-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.yielding_n = cv2.resize(self.yielding_n, (map_width, map_height))
            self.yielding_e = cv2.imread(os.path.join(yielding_area_map_dir, 'yielding_e-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.yielding_e = cv2.resize(self.yielding_e, (map_width, map_height))
            self.yielding_s = cv2.imread(os.path.join(yielding_area_map_dir, 'yielding_s-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.yielding_s = cv2.resize(self.yielding_s, (map_width, map_height))
            self.yielding_w = cv2.imread(os.path.join(yielding_area_map_dir, 'yielding_w-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.yielding_w = cv2.resize(self.yielding_w, (map_width, map_height))

        if at_circle_lane_map_dir is not None:
            self.circle_inner_lane = cv2.imread(os.path.join(at_circle_lane_map_dir, 'circle_inner_lane-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.circle_inner_lane = cv2.resize(self.circle_inner_lane, (map_width, map_height))
            self.circle_outer_lane = cv2.imread(os.path.join(at_circle_lane_map_dir, 'circle_outer_lane-map.jpg'), cv2.IMREAD_GRAYSCALE)
            self.circle_outer_lane = cv2.resize(self.circle_outer_lane, (map_width, map_height))

    def region_position_matching(self, pxl_pt):
        region_position = 'offroad'
        x0, y0 = pxl_pt[0], pxl_pt[1]

        # circle in different quadrant
        if self.circle_1_q_map[y0, x0] > 128.:
            region_position = 'circle_1_q'
            return region_position
        if self.circle_2_q_map[y0, x0] > 128.:
            region_position = 'circle_2_q'
            return region_position
        if self.circle_3_q_map[y0, x0] > 128.:
            region_position = 'circle_3_q'
            return region_position
        if self.circle_4_q_map[y0, x0] > 128.:
            region_position = 'circle_4_q'
            return region_position

        # Entrance. facing circle, left lane is 1, right lane is 2.
        # north entrance
        if self.entrance_n_1[y0, x0] > 128.:
            region_position = 'entrance_n_1'
            return region_position
        if self.entrance_n_2[y0, x0] > 128.:
            region_position = 'entrance_n_2'
            return region_position
        if self.entrance_n_rightturn[y0, x0] > 128.:
            region_position = 'entrance_n_rightturn'
            return region_position
        # east entrance
        if self.entrance_e_1[y0, x0] > 128.:
            region_position = 'entrance_e_1'
            return region_position
        if self.entrance_e_2[y0, x0] > 128.:
            region_position = 'entrance_e_2'
            return region_position
        # south entrance
        if self.entrance_s_1[y0, x0] > 128.:
            region_position = 'entrance_s_1'
            return region_position
        if self.entrance_s_2[y0, x0] > 128.:
            region_position = 'entrance_s_2'
            return region_position
        if self.entrance_s_rightturn is not None:
            if self.entrance_s_rightturn[y0, x0] > 128.:
                region_position = 'entrance_s_rightturn'
                return region_position
        # west entrance
        if self.entrance_w_1[y0, x0] > 128.:
            region_position = 'entrance_w_1'
            return region_position
        if self.entrance_w_2[y0, x0] > 128.:
            region_position = 'entrance_w_2'
            return region_position

        # Exit
        if self.exit_n[y0, x0] > 128.:
            region_position = 'exit_n'
            return region_position
        if self.exit_n_rightturn is not None:
            if self.exit_n_rightturn[y0, x0] > 128.:
                region_position = 'exit_n_rightturn'
                return region_position
        if self.exit_e[y0, x0] > 128.:
            region_position = 'exit_e'
            return region_position
        if self.exit_s[y0, x0] > 128.:
            region_position = 'exit_s'
            return region_position
        if self.exit_s_rightturn is not None:
            if self.exit_s_rightturn[y0, x0] > 128.:
                region_position = 'exit_s_rightturn'
                return region_position
        if self.exit_w[y0, x0] > 128.:
            region_position = 'exit_w'
            return region_position

        # crosswalk
        # if self.crosswalk[y0, x0] > 128.:
        #     region_position = 'crosswalk'
        #     return region_position

        return region_position

    def yielding_area_matching(self, pxl_pt):
        yielding_area = 'Not_in_yielding_area'
        x0, y0 = pxl_pt[0], pxl_pt[1]
        if self.yielding_n[y0, x0] > 128.:
            yielding_area = 'yielding_n'
        if self.yielding_e[y0, x0] > 128.:
            yielding_area = 'yielding_e'
        if self.yielding_s[y0, x0] > 128.:
            yielding_area = 'yielding_s'
        if self.yielding_w[y0, x0] > 128.:
            yielding_area = 'yielding_w'
        return yielding_area

    def at_circle_lane_matching(self, pxl_pt):
        at_circle_lane = 'Not_in_circle'
        x0, y0 = pxl_pt[0], pxl_pt[1]
        if self.circle_inner_lane[y0, x0] > 128.:
            at_circle_lane = 'inner'
            return at_circle_lane
        if self.circle_outer_lane[y0, x0] > 128.:
            at_circle_lane = 'outer'
            return at_circle_lane
        return at_circle_lane

    def entrance_lane_matching(self, pxl_pt):
        """
        This function is to find whether a give position is in certain entrance lane.
        Used for safty check when initializing vehicles.
        """
        # Entrance. facing circle, left lane is 1, right lane is 2.
        # north entrance
        region_position = 'not_at_entrance'
        x0, y0 = pxl_pt[0], pxl_pt[1]

        if self.entrance_n_1[y0, x0] > 128.:
            region_position = 'entrance_n_1'
            return region_position
        if self.entrance_n_2[y0, x0] > 128.:
            region_position = 'entrance_n_2'
            return region_position
        if self.entrance_n_rightturn[y0, x0] > 128.:
            region_position = 'entrance_n_rightturn'
            return region_position
        # east entrance
        if self.entrance_e_1[y0, x0] > 128.:
            region_position = 'entrance_e_1'
            return region_position
        if self.entrance_e_2[y0, x0] > 128.:
            region_position = 'entrance_e_2'
            return region_position
        # south entrance
        if self.entrance_s_1[y0, x0] > 128.:
            region_position = 'entrance_s_1'
            return region_position
        if self.entrance_s_2[y0, x0] > 128.:
            region_position = 'entrance_s_2'
            return region_position
        if self.entrance_s_rightturn is not None:
            if self.entrance_s_rightturn[y0, x0] > 128.:
                region_position = 'entrance_s_rightturn'
                return region_position
        # west entrance
        if self.entrance_w_1[y0, x0] > 128.:
            region_position = 'entrance_w_1'
            return region_position
        if self.entrance_w_2[y0, x0] > 128.:
            region_position = 'entrance_w_2'
            return region_position

        return region_position

if __name__ == '__main__':
    import itertools
    import copy
    import matplotlib.pyplot as plt

    drivable_map_dir = r'ROIs-map/rounD-drivablemap.jpg'
    sim_remove_vehicle_area_map = r'./ROIs-map/rounD-sim-remove-vehicle-area-map.jpg'
    map_height, map_width = 936, 1678
    ROI_matcher = ROIMatcher(drivable_map_dir=drivable_map_dir, sim_remove_vehicle_area_map_dir=sim_remove_vehicle_area_map, map_height=map_height, map_width=map_width)

    drivable_map_check = copy.deepcopy(ROI_matcher.sim_remove_vehicle_area_map)

    for pxl_x, pxl_y in itertools.product(range(map_width), range(map_height)):
        on_road_flag = ROI_matcher.sim_remove_vehicle_area_map[pxl_y, pxl_x] > 128.  # color is white => on the road
        drivable_map_check[pxl_y, pxl_x] = 255 if on_road_flag else 0.

    # subplot(r,c) provide the no. of rows and columns
    plt.figure(figsize=(32, 9))
    f, axarr = plt.subplots(1, 2)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(ROI_matcher.sim_remove_vehicle_area_map)
    plt.title('original map')
    axarr[1].imshow(drivable_map_check)
    plt.title('check map')
    plt.show()