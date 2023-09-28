import numpy as np
import cv2
import os
from trajectory_pool import TrajectoryPool
from geo_engine import GeoEngine
from . import utils

def _draw_vehicle_as_point_with_meter(vis, ptc, color, x_pixel_per_meter, r_meter=1):
    r = int(r_meter * x_pixel_per_meter)

    # draw circle boundary
    cv2.circle(vis, tuple(ptc), r, color, -1)
    # draw center location
    cv2.circle(vis, tuple(ptc), int(r/2), (255, 255, 0), -1)


def _draw_vehicle_as_box(vis, pts, color):
    # fill rectangle
    cv2.fillPoly(vis, [np.array(pts)], color, lineType=cv2.LINE_AA)


def _draw_predicted_future(vis, pts, color):
    # draw mean trajectory
    for i in range(len(pts)-1):
        pt1 = (int(pts[i][0]), int(pts[i][1]))
        pt2 = (int(pts[i+1][0]), int(pts[i+1][1]))
        cv2.line(vis, pt1=pt1, pt2=pt2, color=color, thickness=1, lineType=cv2.LINE_AA)


def _draw_trust_region(vis, pts, r, color):
    # draw trust regions
    for i in range(len(pts)):
        ptc = (int(pts[i][0]), int(pts[i][1]))
        rx, ry = int(r[i][0]), int(r[i][1]) # trust region
        cv2.ellipse(vis, center=ptc, axes=(rx, ry), angle=0, startAngle=0, endAngle=360,
                    color=color, thickness=1, lineType=cv2.LINE_AA)


def _print_vehicle_info(vis, ptc, v, color):
    return

    # print latitude (x)
    pt = (ptc[0] + 15, ptc[1])
    text = "%.6f" % v.location.x
    cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
    # print longitude (y)
    pt = (ptc[0] + 15, ptc[1] + 20)
    text = "%.6f" % v.location.y
    cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
    # print v id
    pt = (ptc[0] + 15, ptc[1] + 40)
    text = "%s" % str(v.id)
    cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)


def _rotate_image(image, angle):
    """
    angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


class Basemap(GeoEngine):
    """
    Basemap for visualization. Show and map detection to base map layer.
    Pixel image: vehicle bounding box, vehicle id
    Map layer: location, heading, rect box, trajectory...
    """

    def __init__(self, map_file_dir, map_height=1024, map_width=1024):
        super(Basemap, self).__init__(map_file_dir, map_height=map_height, map_width=map_width)

        basemap = cv2.imread(map_file_dir, cv2.IMREAD_COLOR)
        self.basemap = cv2.cvtColor(basemap, cv2.COLOR_BGR2RGB)
        self.basemap = cv2.resize(self.basemap, (map_width, map_height))
        self.basemap = (self.basemap.astype(np.float64) * 0.6).astype(np.uint8)

        np.random.seed(0)
        self.color_table = np.random.randint(80, 255, (10, 3))
        np.random.seed()

        self.trajs_manager = TrajectoryPool()
        self.traj_layer = np.zeros([self.h, self.w, 3], dtype=np.uint8)
        self.traj_alpha = np.zeros([self.h, self.w, 3], dtype=np.float32)

        self.get_pixel_resolution()

    def draw_location(self, vehicle_list, color_vid_list=None, rotate_deg=0.):

        if len(vehicle_list) == 0:
            return np.copy(_rotate_image(self.basemap, rotate_deg))
            # return self.basemap

        # vis = np.copy(self.basemap)
        vis = np.copy(_rotate_image(self.basemap, rotate_deg))

        for i in range(0, len(vehicle_list)):
            v = vehicle_list[i]

            if v.location.x is None or v.location.y is None:
                continue

            if color_vid_list is None:
                color = self.color_table[int(v.id) % 10].tolist()
            else:
                if v.id in color_vid_list:
                    # color_list = [[252, 127, 197], [147, 183, 89], [30, 144, 255], [255, 140, 0]]  # red, green, blue, yellow
                    # color_list = [[252, 127, 197], [30, 144, 255], [147, 183, 89], [255, 140, 0]]  # red, blue, green, yellow
                    color_list = [[252, 0, 0], [30, 144, 255], [30, 144, 255], [30, 144, 255], [30, 144, 255], [30, 144, 255]]  # red, blue, green, yellow
                    color = color_list[color_vid_list.index(v.id)]
                else:
                    color = [192, 192, 192]  # grey

            ptc = self._world2pxl([v.location.x, v.location.y])

            if v.realworld_4_vertices is None:
                # box unavailiable, draw a circle instead
                _draw_vehicle_as_point_with_meter(vis, ptc, color, x_pixel_per_meter=self.x_pixel_per_meter, r_meter=1)
            else:
                # box availiable
                pts = self._world2pxl(v.realworld_4_vertices)
                _draw_vehicle_as_box(vis, pts, color)

            if hasattr(v, 'predicted_future') and v.predicted_future is not None:

                location_mean = v.predicted_future['mean']
                location_std = v.predicted_future['std']
                location_3sigma = location_mean + 3*location_std

                pts_mean = self._world2pxl(location_mean)
                pts_3sigma = self._world2pxl(location_3sigma)

                r = np.abs(pts_3sigma - pts_mean)
                _draw_predicted_future(vis, pts_mean, color)
                _draw_trust_region(vis, pts_mean, r, color)

            # print vehicle info beside box
            _print_vehicle_info(vis, ptc, v, (255, 255, 0))

        return vis

    def draw_trajectory(self, vehicle_list, linewidth=2, color_vid_list=None):

        # update trajectory pool
        self.trajs_manager.update(vehicle_list)

        # update alpha (fade out)
        self.traj_alpha *= 0.8

        # draw trajectory
        for vid, value in self.trajs_manager.pool.items():
            if value['update']:

                v = value['vehicle'][-1]
                pt_map = self._world2pxl([v.location.x, v.location.y])
                v_ = value['vehicle'][-2]
                pt_map_prev = self._world2pxl([v_.location.x, v_.location.y])

                pt1 = (int(pt_map[0]), int(pt_map[1]))
                pt2 = (int(pt_map_prev[0]), int(pt_map_prev[1]))

                if color_vid_list is None:
                    color = self.color_table[int(v.id) % 10].tolist()
                else:
                    if v.id in color_vid_list:
                        # color_list = [[252, 127, 197], [147, 183,  89], [30, 144, 255], [255, 140, 0]]  # red, green, blue, yellow
                        # color_list = [[252, 127, 197], [30, 144, 255], [147, 183, 89], [255, 140, 0]]  # red, blue, green, yellow
                        color_list = [[252, 0, 0], [30, 144, 255], [30, 144, 255], [30, 144, 255], [30, 144, 255], [30, 144, 255]]  # red, blue, green, yellow
                        color = color_list[color_vid_list.index(v.id)]
                    else:
                        color = [192, 192, 192]  # grey

                cv2.line(self.traj_layer, pt1=pt1, pt2=pt2, color=color, thickness=linewidth)
                cv2.line(self.traj_alpha, pt1=pt1, pt2=pt2, color=(0.8, 0.8, 0.8), thickness=linewidth)

        return self.traj_layer, self.traj_alpha

    @ staticmethod
    def layer_blending(base_layer, traj_layer, traj_alpha):
        base_layer = base_layer.astype(np.float32)/255.
        traj_layer = traj_layer.astype(np.float32)/255.
        vis = base_layer*(1-traj_alpha) + traj_layer*traj_alpha
        return vis

    def render(self, vehicle_list, with_traj=True, linewidth=2, color_vid_list=None, rotate_deg=0.):
        """
        rotate_deg – Rotate the background map. Rotation angle in degrees. Positive values mean counter-clockwise rotation
        """
        base_layer = self.draw_location(vehicle_list, color_vid_list, rotate_deg=rotate_deg)
        if with_traj:
            traj_layer, traj_alpha = self.draw_trajectory(vehicle_list, linewidth, color_vid_list)
            map_vis = self.layer_blending(base_layer, traj_layer, traj_alpha)
            map_vis = (map_vis*255.).astype(np.uint8)
        else:
            map_vis = base_layer

        return map_vis

    def draw_location_vis_grad(self, vehicle_list, current_t_idx=None, ego_vid=None, grad_res={}, color_vid_list=None):

        if len(vehicle_list) == 0:
            return self.basemap

        vis = np.copy(self.basemap).astype(np.float32)
        vis = vis / 255

        # create a new layer and its alpha matte
        new_layer = np.zeros_like(self.basemap.astype(np.float32))
        alpha_matte = np.zeros_like(self.basemap.astype(np.float32))

        for i in range(0, len(vehicle_list)):
            v = vehicle_list[i]

            if v.location.x is None or v.location.y is None:
                continue

            if v.id == ego_vid:
                # color = (30, 144, 255)  # blue
                if v.id in grad_res.keys():
                    percentage = grad_res[v.id]
                else:
                    percentage = 0.5
                # alpha = [1.5 * percentage for i in range(3)]
                alpha = (1.0, 1.0, 1.0)
                color = (0, 0, 1)  # blue

            elif v.id in grad_res.keys():
            # if v.id in grad_res.keys():
                percentage = grad_res[v.id]
                color = (1, 0, 0)  # red
                # color = (0.752, 0.752, 0.752)  # grey
                alpha = [percentage * 2 for i in range(3)]
                # alpha = [min(1, percentage * 3) for i in range(3)]
                # alpha = (1.0, 1.0, 1.0)
            else:
                color = (0.752, 0.752, 0.752)  # grey
                alpha = (1.0, 1.0, 1.0)

            ptc = self._world2pxl([v.location.x, v.location.y])
            pts = self._world2pxl(v.realworld_4_vertices)
            # _draw_vehicle_as_box(vis, pts, color)
            cv2.fillPoly(new_layer, [np.array(pts)], color, lineType=cv2.LINE_AA)
            cv2.fillPoly(alpha_matte, [np.array(pts)], alpha, lineType=cv2.LINE_AA)

            # print vehicle info beside box
            _print_vehicle_info(vis, ptc, v, (0.752, 0.752, 0.752))

        # Plot line connecting the ego-vehicle and other vehicles
        for i in range(0, len(vehicle_list)):
            v = vehicle_list[i]
            if v.id == ego_vid:
                ego_v_ptc = self._world2pxl([v.location.x, v.location.y])
                for j in range(0, len(vehicle_list)):
                    other_v = vehicle_list[j]
                    other_v_ptc = self._world2pxl([other_v.location.x, other_v.location.y])
                    if other_v.id in grad_res.keys():
                        percentage = grad_res[other_v.id]
                        color = (1, 1, 0)  # yellow
                        # color = (1, 0, 0)  # red
                        alpha = [percentage * 2 for i in range(3)]
                        cv2.line(new_layer, tuple(ego_v_ptc), tuple(other_v_ptc), color, thickness=2, lineType=cv2.LINE_AA)
                        cv2.line(alpha_matte, tuple(ego_v_ptc), tuple(other_v_ptc), alpha, thickness=2, lineType=cv2.LINE_AA)

        # write current time idx on the frame
        if current_t_idx is not None:
            pt = (300, 80)
            text = "%s" % str(current_t_idx)
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2.0, color=(0.752, 0.752, 0.752), thickness=2, lineType=cv2.LINE_AA)

        vis = vis * (1 - alpha_matte) + new_layer * alpha_matte

        return vis

    def get_pixel_resolution(self):
        dx_meter, dy_meter = self.f.tl[0] - self.f.tr[0], self.f.tl[1] - self.f.tr[1]
        d = np.linalg.norm([dx_meter, dy_meter])

        self.x_pixel_per_meter = self.w / d

        dx_meter, dy_meter = self.f.tl[0] - self.f.bl[0], self.f.tl[1] - self.f.bl[1]
        d = np.linalg.norm([dx_meter, dy_meter])

        self.y_pixel_per_meter = self.h / d
