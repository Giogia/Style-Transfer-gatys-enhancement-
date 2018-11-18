import tensorflow as tf


WEIGHTS_INIT_STDEV = .1


def net(image):

    conv1 = conv_layer(image, 32, 9, 1)
    conv2 = conv_layer(conv1, 64, 3, 2)
    conv3 = conv_layer(conv2, 128, 3, 2)
    resid1 = residual_block(conv3, 3)
    resid2 = residual_block(resid1, 3)
    resid3 = residual_block(resid2, 3)
    resid4 = residual_block(resid3, 3)
    resid5 = residual_block(resid4, 3)
    conv_t1 = conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = conv_layer(conv_t2, 3, 9, 1, is_relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2

    return preds


def conv_layer(image, filter_number, filter_size, strides, is_relu=True):

    # make the convolution of the image and return the convolution

    weights_initialization = conv_initialization_vars(image, filter_number, filter_size)
    strides_shape = [1, strides, strides, 1]

    # apply the filter to the image with a 2d convolution
    image = tf.nn.conv2d(image, weights_initialization, strides_shape, padding='SAME')

    image = _instance_norm(image)

    if is_relu:

        image = tf.nn.relu(image)

    return image


def conv_tranpose_layer(img, filter_number, filter_size, strides):

    weights_initialized = conv_initialization_vars(img, filter_number, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i for i in img.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    new_shape = [batch_size, new_rows, new_cols, filter_number]

    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    convolution = tf.nn.conv2d_transpose(img, weights_initialized, tf_shape, strides_shape, padding='SAME')
    convolution = _instance_norm(convolution)

    return tf.nn.relu(convolution)


def residual_block(img, filter_size=3):

    tmp_convolution = conv_layer(img, 128, filter_size, 1)

    # add the convolution to the original image
    return img + conv_layer(tmp_convolution, 128, filter_size, 1, is_relu=False)


def _instance_norm(img, train=True):

    # set the shape of the input img
    batch_size, rows, cols, in_channels = [i for i in img.get_shape()]

    var_shape = [in_channels]

    # calculate the mean and the variance of the img
    mu, sigma_sq = tf.nn.moments(img, [1, 2], keep_dims=True)

    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3

    # normalize the img input wrt the mean and the variance calculated
    normalized = (img - mu) / (sigma_sq + epsilon) ** (.5)

    return scale * normalized + shift


def conv_initialization_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:

        weights_shape = [filter_size, filter_size, in_channels, out_channels]

    else:

        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    # with tf truncated we output rnd values rom a truncated normal distribution
    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)

    return weights_init






