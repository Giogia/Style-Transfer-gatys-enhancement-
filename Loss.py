import tensorflow as tf


def g_matrix(tensor):

    channels = int(tensor.shape[-1])

    # reshape as 1-Dim array dividing it per channel
    a = tf.reshape(tensor, [-1, channels])

    # compute the matrix a*a^t and then divide by the dimension
    return tf.matmul(a, a, transpose_a=True) / tf.cast(tf.shape(a)[0], tf.float32)



"""If one or both of the matrices contain a lot of zeros, a more efficient multiplication 
algorithm can be used by setting the corresponding a_is_sparse or b_is_sparse flag to True"""


def get_content_loss(content, target):

    return tf.reduce_mean(tf.square(content - target))


def get_style_loss(style, g_target):

    g_style = g_matrix(style)
    height, width, channels = list(style.get_shape())
    #weight = (4. * (channels ** 2) * (width * height) ** 2)

    return tf.reduce_mean(tf.square(g_style - g_target)) #/ weight


def accumulate_loss(img_feature, layers_n, noise_feature, loss):

    score = 0
    weight_per_layer = 1.0 / float(layers_n)

    for target, comb_content in zip(img_feature, noise_feature):
        score += weight_per_layer * loss(comb_content, target)

    return score


def compute_loss(noise_features, img_features, loss_w, layers_n):
    """This function will compute the loss total loss.

    Arguments:
      noise_features: The content and style of the white noise
      loss_w: The weights of each contribution of each loss function.
        (style weight, content weight, and total variation weight)
      img_features: Content and style features
      loss_w: Weights of the elements
      layers_n: Number of content and style layers

    Returns:
      returns the total loss
    """

    # Accumulate content losses from all layers
    content_score = accumulate_loss(img_features[0], layers_n[0], noise_features[0], get_content_loss)

    # Accumulate style losses from all layers
    style_score = accumulate_loss(img_features[1], layers_n[1], noise_features[1], get_style_loss)

    # Here, we equally weight each contribution of each loss layer
    content_score *= loss_w[0]
    style_score *= loss_w[1]

    return style_score + content_score


def compute_gradient(noise_img, noise_features_gen, img_features, loss_w, layers_n):

    with tf.GradientTape() as g:
        loss = compute_loss(noise_features_gen(noise_img), img_features, loss_w, layers_n)

    # Compute gradients wrt input image
    return g.gradient(loss, noise_img), loss
