import gzip
import pickle

import numpy as np

"""
TODO: 
Same as HW1. Feel free to copy and paste your old implementation here.
It's a good time to vectorize it, while you're at it!
No need to include CIFAR-specific methods.
"""

def get_data_MNIST(subset, data_path="../data"):
    """
    :param subset: string indicating whether we want the training or testing data 
        (only accepted values are 'train' and 'test')
    :param data_path: directory containing the training and testing inputs and labels
    :return: NumPy array of inputs (float32) and labels (uint8)
    """
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"
    image = []
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        for i in range(num_examples):
            img = bytestream.read(784) # read the image
            img = np.frombuffer(img, dtype=np.uint8)
            img = np.float32(img/255.0) # normalize
            img = img.reshape(784,) #flatted
            image.append(img)
    image = np.asarray(image)
    
    label = []
    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        for j in range(num_examples):
            lbl = bytestream.read(1)
            lbl = np.frombuffer(lbl, dtype=np.uint8)
            label.append(lbl[0])
    label = np.asarray(label)
    return image, label
    
## THE REST ARE OPTIONAL!


def shuffle_data(image_full, label_full, seed):
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full

def get_specific_class(image_full, label_full, specific_class=0, num=None):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a specific digit.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image_full: the image array returned by the get_data function
    :param label_full: the label array returned by the get_data function
    :param specific_class: the specific class you want
    :param num: number of the images and labels to return
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """
    mask = np.ma.masked_where(label_full == specific_class, label_full) # create a mask
    label = label_full[mask.mask] 
    image = image_full[mask.mask]
    image = image[:num] 
    label = label[:num]

    return image, label


def get_subset(image_full, label_full, class_list, num):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.
    Also, for KNN, we only need a small subset of the dataset.
    So, we need a function that selects the images and labels for a list of specific digits.

    The same for the CIFAR dataset. We only need a small subset of CIFAR.

    :param image: the image array returned by the get_data function
    :param label: the label array returned by the get_data function
    :param class_list: the list of specific classes you want
    :param num: number of the images and labels to return for each class
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """

    image_list = []
    label_list = []
    for cl in class_list: # loop through each class 
        images, labels = get_specific_class(image_full, label_full, cl,num)
        image_list.append(images)
        label_list.append(labels)
    image = np.concatenate(image_list)
    label = np.concatenate(label_list)

    

    return image, label


