import numpy as np
import cv2
import json
import os


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_config(path_to_json=r'./config.json'):
    with open(path_to_json) as f:
      data = json.load(f)
    return Struct(**data)


class GeoEngine(object):
    """
    Build Homography between pixel map and realworld coordinate.
    Base class for basemap visualizer.
    """

    def __init__(self, map_file_dir, map_height=1024, map_width=1024):

        self.h, self.w = map_height, map_width

        self.f = parse_config(os.path.splitext(map_file_dir)[0] + '.json')
        self.transform_wd2px, self.transform_px2wd = self._create_coord_mapper()

    def _create_coord_mapper(self):

        # wd_tl, wd_tr, wd_bl, wd_br = self.f.tl.copy(), self.f.tr.copy(), self.f.bl.copy(), self.f.br.copy()
        px_tl, px_tr, px_bl, px_br = [0, 0], [self.w-1, 0], [0, self.h-1], [self.w-1, self.h-1]

        # normalize to local coordination
        wd_tl = (self.f.tl[0], self.f.tl[1])
        wd_tr = (self.f.tr[0], self.f.tr[1])
        wd_bl = (self.f.bl[0], self.f.bl[1])
        wd_br = (self.f.br[0], self.f.br[1])

        wd_points = np.array([wd_tl, wd_tr, wd_bl, wd_br], np.float32)
        px_points = np.array([px_tl, px_tr, px_bl, px_br], np.float32)

        transform_wd2px = cv2.getPerspectiveTransform(src=wd_points, dst=px_points)
        transform_px2wd = cv2.getPerspectiveTransform(src=px_points, dst=wd_points)

        return transform_wd2px, transform_px2wd

    def _world2pxl(self, pt_world, output_int=True):

        # normalize to local coordination
        pt_world = np.array(pt_world).reshape([-1, 2])
        lat, lon = pt_world[:, 0], pt_world[:, 1]
        lat_norm, lon_norm = lat, lon
        pt_world = np.array([lat_norm, lon_norm]).T

        pt_world = np.array(pt_world).reshape([-1, 1, 2]).astype(np.float32)
        pt_pixel = cv2.perspectiveTransform(pt_world, self.transform_wd2px)
        pt_pixel = np.squeeze(pt_pixel)

        if output_int:
            pt_pixel = pt_pixel.astype(np.int)

        return pt_pixel

    def _pxl2world(self, pt_pixel):

        pt_pixel = np.array(pt_pixel).reshape([-1, 1, 2]).astype(np.float32)
        pt_world = cv2.perspectiveTransform(pt_pixel, self.transform_px2wd).astype(np.float64)
        pt_world = np.squeeze(pt_world)
        pt_world = pt_world.reshape([-1, 2])

        # unnormalize to world coordination
        lat_norm, lon_norm = pt_world[:, 0], pt_world[:, 1]
        lat, lon = lat_norm, lon_norm
        pt_world = np.array([lat, lon]).T

        return pt_world

