"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE

    # unzip file and check header
    with gzip.open(image_filename, "rb") as f:
        image_bytes = f.read()
    image_header = struct.unpack(">4i", image_bytes[:16])
    image_magic_number, image_n, row_n, col_n = image_header
    if image_magic_number != 2051:
        raise Exception("Magic Number Check Failure")
    print(f"image_header: {image_header}")

    # process content
    input_dim = row_n * col_n
    image_content = image_bytes[16:]
    X = np.frombuffer(image_content, dtype=np.uint8).astype(np.float32).reshape(image_n, input_dim) / 255

    # unzip file and check header
    with gzip.open(label_filename, "rb") as f:
        label_bytes = f.read()
    label_header = struct.unpack(">2i", label_bytes[:8])
    label_magic_number, label_n= label_header
    if label_magic_number != 2049:
        raise Exception("Magic Number Check Failure")
    print(f"label_header: {label_header}")

    # process content
    label_content = label_bytes[8:]
    y = np.frombuffer(label_content, dtype=np.uint8)

    return X, y

    ### END YOUR CODE


def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION

    total_loss = ndl.log(ndl.summation(ndl.exp(Z), axes=(1, ))) - ndl.summation(Z * y_one_hot, axes=(1, ))
    return total_loss.sum() / Z.shape[0]

    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    start_index = 0
    k = W2.shape[1]
    while start_index < len(X):
        batch_size = batch if start_index + batch <= len(X) else len(X) - start_index
        x_mini_batch = ndl.Tensor(X[start_index: start_index + batch_size])
        y_mini_batch = ndl.Tensor(y[start_index: start_index + batch_size])

        h = ndl.relu(x_mini_batch @ W1) @ W2
        i_y = ndl.Tensor(np.eye(k)[y_mini_batch.numpy()])

        loss = softmax_loss(h, i_y)
        loss.backward()

        # set with detached data
        W1.data = W1.data - lr * W1.grad.data
        W2.data = W2.data - lr * W2.grad.data

        start_index += batch_size
    return W1, W2
    ### END YOUR SOLUTION


    # ### BEGIN YOUR CODE

    # num_examples, _ = X.shape
    # _, num_classes = W2.shape
    # print(f"X.shape = {X.shape}")
    # print(f"W2.shape = {W2.shape}")

    # n_batch = num_examples // batch
    # for i in range(n_batch):
    #     batch_start = i * batch
    #     batch_end = batch_start + batch
    #     X_batch = X[batch_start : batch_end]
    #     y_batch = y[batch_start : batch_end]

    #     X_batch = ndl.Tensor.make_const(X_batch)
    #     y_one_hot_batch = np.zeros((len(y_batch), num_classes), dtype=np.uint8)
    #     y_one_hot_batch[np.arange(len(y_batch)), y_batch] = 1
    #     y_one_hot_batch = ndl.Tensor.make_const(y_one_hot_batch)

    #     # construct computational graph
    #     logits = ndl.relu(X_batch @ W1) @ W2
    #     loss = softmax_loss(logits, y_one_hot_batch)

    #     # update weights
    #     loss.backward()
    #     W1 = W1.detach() - lr * W1.grad.detach()
    #     W2 = W2.detach() - lr * W2.grad.detach()

    # return W1, W2

    # ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
