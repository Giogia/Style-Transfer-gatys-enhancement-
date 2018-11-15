import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import CNN
import Image
import Loss


def run_style_transfer(content_path, style_path, iterations=1000, content_weight=1e-1, style_weight=1e9, learning_rate=5):

    #create images
    content = Image.load_image(content_path)
    style = Image.load_image(style_path)
    noise = Image.generate_noise_image(content)

    content = Image.preprocess_image(content)
    style = Image.preprocess_image(style)
    noise = Image.preprocess_image(noise)

    noise = tfe.Variable(content, dtype=tf.float32)

    # create model
    vgg = CNN.VGG19_c()
    loss_weights = content_weight, style_weight
    layers_number = vgg.content_layers_num , vgg.style_layers_num

    #create features
    content_features = vgg.get_content_features(content)
    style_features = vgg.get_style_features(style)
    gram_matrix_features = [Loss.g_matrix(feature) for feature in style_features]

    img_features = content_features, gram_matrix_features

    #create optimizer
    opt = tf.train.AdamOptimizer(learning_rate, beta1=0.99, epsilon=1e-1)

    #store best results
    best_loss, best_img = float('inf'), None

    #plt.ion()
    for i in range(iterations):

        print(i)

        grads, loss = Loss.compute_gradient(noise,vgg.get_output_features,img_features,loss_weights,layers_number)

        opt.apply_gradients([(grads, noise)])

        clipped = Image.clip_image(noise)
        noise.assign(clipped)


        if loss < best_loss:

            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = Image.postprocess_image(noise.numpy())
        if i %10 == 0:
            print(loss,"best",best_loss)
            plot_img = noise.numpy()
            plot_img = Image.postprocess_image(plot_img)
            Image.show_image(plot_img)

        plt.show()

    Image.save_image('/Images/Result.jpg',best)

    return best_loss, best_img


if __name__ == "__main__":

    tf.enable_eager_execution()
    print("Eager execution: {}".format(tf.executing_eagerly()))

    content_path = '/Images/Eiffel.jpg'
    style_path = '/Images/VanGogh.jpg'

    Image.show_content_style(content_path, style_path)

    best, best_loss = run_style_transfer(content_path, style_path, iterations=1000)