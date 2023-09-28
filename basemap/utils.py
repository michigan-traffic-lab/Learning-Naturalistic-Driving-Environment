import numpy as np
import cv2
import json


############ some helper functions ##############

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_config(path_to_json=r'./config.json'):
    with open(path_to_json) as f:
      data = json.load(f)
    return Struct(**data)



################ BB draw on 2D image... ####################

def draw_bb_on_image(vehicle_list, img):

    for i in range(len(vehicle_list)):
        v = vehicle_list[i]
        x_bottom_c, y_bottom_c = v.pixel_bottom_center.x, v.pixel_bottom_center.y
        diagonal_length = v.diagonal_length_pixel

        # draw box
        pt1 = (int(x_bottom_c - diagonal_length / 2), int(y_bottom_c - diagonal_length / 2))
        pt2 = (int(x_bottom_c + diagonal_length / 2), int(y_bottom_c + diagonal_length / 2))
        cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        box_tmp = img[pt1[1]:pt2[1], pt1[0]:pt2[0], :]
        img[pt1[1]:pt2[1], pt1[0]:pt2[0], 1:] = (0.75 * box_tmp[:, :, 1:]).astype(np.uint8)

        # draw bottom center
        cv2.circle(img, (x_bottom_c, y_bottom_c), 2, (255, 0, 0), -1)
        text = 'id=%s' % v.id
        pt = (x_bottom_c, y_bottom_c - 5)
        cv2.putText(img, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    return img

