import numpy as np


def xyxy_to_xywh(xyxy):
    """Calculates the relative bounding box from absolute pixel values."""
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return [x_c, y_c, w, h]


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] 
    to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def tlwh2xyxy(x):
    """" Convert tlwh to xyxy """
    y = np.copy(x)
    y[:, 2] = x[:, 2] + x[:, 0]
    y[:, 3] = x[:, 3] + x[:, 1]
    return y


def xyxy2tlwh(x):
    """" Convert xyxy to tlwh """
    y = np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def bbox_cxcywh_to_xyxy(x):
    cx = x[:, 0]
    cy = x[:, 1]
    _w = x[:, 2]
    _h = x[:, 3]
    x1 = cx - 0.5 * _w
    y1 = cy - 0.5 * _h
    x2 = cx + 0.5 * _w
    y2 = cy + 0.5 * _h
    return np.stack([x1, y1, x2, y2], axis=1)


def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate(
        [w/2 * Cos, -w/2 * Sin], axis=-1)
    vector2 = np.concatenate(
        [-h/2 * Sin, -h/2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    order = obboxes.shape[:-1]
    return np.concatenate(
        [point1, point2, point3, point4], axis=-1).reshape(*order, 8)