import tensorflow as tf
from pennylane import numpy as np
import matplotlib.pyplot as plt

def get_dataset(digits=[3, 5], n_px=8, train_size=1000, test_size=200):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[...,
                                                               np.newaxis] / 255.0

    # Create a boolean mask to filter out only the samples with desired label
    filter_mask = np.isin(y_train, digits)
    
    # Apply the filter mask to the features and labels to keep only the selected digits
    x_train = x_train[filter_mask]
    y_train = y_train[filter_mask]
    
    # Do the same for test dataset
    filter_mask = np.isin(y_test, digits)
    
    x_test = x_test[filter_mask]
    y_test = y_test[filter_mask]

    # Randomly select samples
    random_indices = np.random.choice(x_train.shape[0], train_size, replace=False)
    
    # Select the corresponding samples
    x_train = x_train[random_indices]
    y_train = y_train[random_indices]  # Select corresponding labels
    
    # Do the same for test samples
    random_indices = np.random.choice(x_test.shape[0], test_size, replace=False)
    
    # Select the corresponding samples
    x_test = x_test[random_indices]
    y_test = y_test[random_indices]  # Select corresponding labels

    with tf.device(':CPU:0'):
        x_train = tf.image.resize(x_train, (n_px, n_px)).numpy()
        x_test = tf.image.resize(x_test, (n_px, n_px)).numpy()

    return (x_train, y_train), (x_test, y_test)

def visualise_data(digits, x, y, pred=None):
    plt.style.use('default')
    n_img = len(x)
    labels_list = digits
    fig, axes = plt.subplots(1, len(x), figsize=(2*len(x), 2))
    for i in range(n_img):
        axes[i].imshow(x[i], cmap="gray")
        if pred is None:
            axes[i].set_title("Label: {}".format(labels_list[y[i]]))
        else:
            axes[i].set_title("Label: {}, Pred: {}".format(labels_list[y[i]], labels_list[pred[i]]))
    plt.tight_layout(w_pad=2)