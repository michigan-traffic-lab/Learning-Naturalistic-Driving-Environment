import numpy as np
from shapely.geometry import Polygon


def get_box_pts_from_center_heading(length, width, xc, yc, heading):

    l, w = length / 2.0 , width / 2.0

    ## box
    x1, y1 = l, w
    x2, y2 = l, -w
    x3, y3 = -l, -w
    x4, y4 = -l, w

    ## rotation
    a = heading / 180. * np.pi
    x1_, y1_ = _rotate_pt(x1, y1, a)
    x2_, y2_ = _rotate_pt(x2, y2, a)
    x3_, y3_ = _rotate_pt(x3, y3, a)
    x4_, y4_ = _rotate_pt(x4, y4, a)

    ## translation
    pt1 = [x1_ + xc, y1_ + yc]
    pt2 = [x2_ + xc, y2_ + yc]
    pt3 = [x3_ + xc, y3_ + yc]
    pt4 = [x4_ + xc, y4_ + yc]

    return [pt1, pt2, pt3, pt4]


def _rotate_pt(x, y, a):
    return np.cos(a)*x - np.sin(a)*y, np.sin(a)*x + np.cos(a)*y


class Location(object):
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z


class Point2d(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


class Point3d(object):
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z


class Size3d(object):
    def __init__(self, width=None, length=None, height=None):
        self.width = width
        self.length = length
        self.height = height


class Rotation(object):
    def __init__(self, yaw=None, pitch=None, roll=None):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


class Vehicle(object):
    def __init__(self):
        "Configuration for vehicle states"
        self.location = Location()
        self.rotation = Rotation()
        self.size = Size3d()
        self.pixel_bottom_center = Point2d()
        self.pixel_bb = None
        self.diagonal_length_pixel = None
        self.pixel_8_vertices = None
        self.realworld_8_vertices = None
        self.realworld_4_vertices = None
        self.id = '-1'
        self.uuid = '-1'
        self.category = None
        self.confidence = None
        self.speed = None
        self.speed_heading = None
        self.predicted_future = None
        self.poly_box = None  # The rectangle of the vehicle using shapely.Polygon
        self.safe_poly_box = None  # The rectangle of the vehicle (with a safety buffer) using shapely.Polygon
        self.safe_size = Size3d()  # Include a buffer compared with vehicle real size
        self.gt_size = Size3d()  # Vehicle real-world gt size (e.g., a truck from rounD). During modeling, we might consider them as another (e.g., identical) size using self.size.
        self.gt_realworld_4_vertices = None  # using gt_size
        self.gt_poly_box = None  # using gt_size

    def update_poly_box_and_realworld_4_vertices(self):
        """
        Update the poly box and realworld 4 vertices based on current location (x,y) and speed heading
        Returns
        -------

        """
        if self.size.length is not None:
            length, width = self.size.length, self.size.width
        else:
            length, width = 3.6, 1.8
        realworld_4_vertices = get_box_pts_from_center_heading(length=length, width=width, xc=self.location.x, yc=self.location.y, heading=self.speed_heading)
        new_poly_box = Polygon(realworld_4_vertices)
        self.poly_box = new_poly_box
        self.realworld_4_vertices = np.array(realworld_4_vertices)

    def update_safe_poly_box(self):
        if self.safe_size.length is not None:
            length, width = self.safe_size.length, self.safe_size.width
        else:
            length, width = 3.8, 2.0
        # length, width = 3.8, 2.0
        new_safe_poly_box = Polygon(get_box_pts_from_center_heading(length=length, width=width, xc=self.location.x, yc=self.location.y, heading=self.speed_heading))
        self.safe_poly_box = new_safe_poly_box
