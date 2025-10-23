import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]


def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))

    # Feature 1: Measure of Symmetry
    # We calculate symmetry by flipping the image horizontally and taking the
    # negative average of the absolute difference. A more symmetric image will have a value closer to 0.
    flipped_images = np.fliplr(raw_image)
    feature_symmetry = -np.mean(np.abs(raw_image - flipped_images), axis=(1, 2))


    # Feature 2: Measure of Intensity
    # This is the average pixel value across the entire 16x16 image.
    feature_intensity = np.mean(raw_X, axis=1)


    # Feature 3: Bias Term. Always 1.
    # A bias term (a column of ones) is added to allow the model to learn an offset,
    # similar to the y-intercept in a linear equation.
    feature_bias = np.ones(len(raw_X))


    # Stack features together in the following order.
    # [Feature 3, Feature 1, Feature 2]
    # We stack them horizontally to create a feature matrix of shape [n_samples, 3].
    X = np.stack([feature_bias, feature_symmetry, feature_intensity], axis=1)

    return X

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx




