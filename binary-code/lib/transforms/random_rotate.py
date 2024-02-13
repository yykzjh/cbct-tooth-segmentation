import numpy as np
import scipy.ndimage as ndimage


def random_rotate3D(img_numpy, label=None, min_angle=-10, max_angle=10):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param label: 3D label array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]

    img_numpy = ndimage.rotate(img_numpy, angle, axes=axes, reshape=False)

    if label is not None:
        if label.ndim == 4:
            for ch in range(label.shape[0]):
                label[ch, :, :, :] = ndimage.rotate(label[ch, :, :, :], angle, axes=axes, reshape=False)
        else:
            label = ndimage.rotate(label, angle, axes=axes, reshape=False, order=0)
        return img_numpy, label

    return img_numpy


class RandomRotation(object):
    def __init__(self, min_angle=-10, max_angle=10):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        if label is None:
            img_numpy = random_rotate3D(img_numpy, min_angle=self.min_angle, max_angle=self.max_angle)
            return img_numpy
        else:
            img_numpy, label = random_rotate3D(img_numpy, label=label, min_angle=self.min_angle, max_angle=self.max_angle)
            return img_numpy, label
