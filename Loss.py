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


def compute_loss(noise_features, img_features, loss_w, layers_n):
    """This function will compute the loss total loss.

    Arguments:
      model: The model that will give us access to the intermediate layers
      loss_w: The weights of each contribution of each loss function.
        (style weight, content weight, and total variation weight)
      init_img: Our initial base image. This image is what we are updating with
        our optimization process. We apply the gradients wrt the loss we are
        calculating to this image.
      g_style_f: Precomputed gram matrices corresponding to the
        defined style layers of interest.
      content_f: Precomputed outputs from defined content layers of
        interest.

    Returns:
      returns the total loss, style loss, content loss, and total variational loss
    """

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!

    # Accumulate style losses from all layers
    # Accumulate content losses from all layers
    content_score = accumulate_loss(img_features[0], layers_n[0], noise_features[0], get_content_loss)

    # Here, we equally weight each contribution of each loss layer
    style_score = accumulate_loss(img_features[1], layers_n[1], noise_features[1], get_content_loss)

    content_score *= loss_w[0]
    style_score *= loss_w[1]
    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


def accumulate_loss(img_feature, layers_n, noise_feature, loss):
    score = 0
    weight_per_layer = 1.0 / float(layers_n)
    for target, comb_content in zip(img_feature, noise_feature):
        score += weight_per_layer * loss(comb_content[0], target)
    return score


def compute_gradient(init_img, noise_features, img_features, loss_w, layers_n):
    with tf.GradientTape() as g:
        all_loss = compute_loss(noise_features, img_features, loss_w, layers_n)
        # Compute gradients wrt input image
    return g.gradient(all_loss[0], init_img), all_loss
