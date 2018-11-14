import tensorflow as tf

def g_matrix(tensor):
    channels = int(tensor.shape()[-1])  # possible to use to static rep?
    a = tf.reshape(tensor, [-1, channels])  # reshape as 1-Dim array dividing it per channel
    return tf.matmul(a, a, True) / tf.cast(tf.shape(a)[0], tf.float32)
    # compute the matrix a*a^t and then divide by the dimension


"""If one or both of the matrices contain a lot of zeros, a more efficient multiplication 
algorithm can be used by setting the corresponding a_is_sparse or b_is_sparse flag to True"""


def get_content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))


def get_style_loss(style, g_target):
    g_style = g_matrix(style)
    # height, width, channels = list(style.get_shape())
    # weight = (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(g_style - g_target))  # / weight
