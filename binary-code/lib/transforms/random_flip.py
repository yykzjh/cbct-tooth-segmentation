import numpy as np


# TODO test

def random_flip(img_numpy, label=None, axis_for_flip=0):
    axes = [0, 1, 2]
    img_numpy = flip_axis(img_numpy, axes[axis_for_flip])
    img_numpy = np.squeeze(img_numpy)

    if label is not None:
        if label.ndim == 4:
            for ch in range(label.shape[0]):
                y = flip_axis(label[ch, :, :, :], axes[axis_for_flip])
                label[ch, :, :, :] = np.squeeze(y)
        else:
            y = flip_axis(label, axes[axis_for_flip])
            label = np.squeeze(y)
        return img_numpy, label

    return img_numpy


def flip_axis(img_numpy, axis):
    img_numpy = np.asarray(img_numpy).swapaxes(axis, 0)
    img_numpy = img_numpy[::-1, ...]
    img_numpy = img_numpy.swapaxes(0, axis)
    return img_numpy


class RandomFlip(object):
    def __init__(self):
        self.axis_for_flip = 0

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be flipped.
            label (numpy): Label segmentation map to be flipped

        Returns:
            img_numpy (numpy):  flipped img.
            label (numpy): flipped Label segmentation.
        """
        self.axis_for_flip = np.random.randint(0, 3)
        if label is None:
            return random_flip(img_numpy, axis_for_flip=self.axis_for_flip)
        else:
            return random_flip(img_numpy, label=label, axis_for_flip=self.axis_for_flip)
