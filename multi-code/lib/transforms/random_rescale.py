import numpy as np
import scipy.ndimage as ndimage


def random_rescale(img_numpy, label=None, min_percentage=0.8, max_percentage=1.1):
    """
    Args:
        img_numpy:
        label:
        min_percentage:
        max_percentage:

    Returns:

    """
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])

    img_numpy = ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)

    if label is not None:
        if label.ndim == 4:
            for ch in range(label.shape[0]):
                label[ch, :, :, :] = ndimage.interpolation.affine_transform(label[ch, :, :, :], zoom_matrix)
        else:
            label = ndimage.interpolation.affine_transform(label, zoom_matrix, order=0)
        return img_numpy, label

    return img_numpy


class RandomRescale(object):
    def __init__(self, min_percentage=0.8, max_percentage=1.1):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        if label is None:
            img_numpy = random_rescale(img_numpy, min_percentage=self.min_percentage, max_percentage=self.max_percentage)
            return img_numpy
        else:
            img_numpy, label = random_rescale(img_numpy, label=label, min_percentage=self.min_percentage, max_percentage=self.max_percentage)
            return img_numpy, label
