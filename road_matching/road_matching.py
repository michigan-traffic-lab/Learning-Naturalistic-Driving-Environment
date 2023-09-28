from geo_engine import GeoEngine
import cv2
import json
import numpy as np
import os
import copy


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_config(path_to_json=r'./config.json'):
    with open(path_to_json) as f:
      data = json.load(f)
    return Struct(**data)


class RoadMatcher(GeoEngine):
    """
    Fast road matcher. Given a drivable map, this module maps any coordinate to
    the closest location within the drivable set.
    """

    def __init__(self, map_file_dir, map_height=1024, map_width=1024):
        super(RoadMatcher, self).__init__(map_file_dir, map_height=map_height, map_width=map_width)

        road_map = cv2.imread(map_file_dir, cv2.IMREAD_GRAYSCALE)
        self.road_map = cv2.resize(road_map, (map_width, map_height))

    def _within_map(self, lat, lon):
        lat_max = self.f.br[0]
        lat_min = self.f.tl[0]
        lon_max = self.f.tl[1]
        lon_min = self.f.br[1]
        if lat < lat_max and lat > lat_min and lon < lon_max and lon > lon_min:
            return True
        else:
            return False

