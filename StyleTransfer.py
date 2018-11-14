import CNN, Image, Loss
import tensorflow.contrib.eager as tfe
import tensorflow as tf
import matplotlib.pyplot as plt

def run_style_transfer(content_path, style_path, iterations=1000, content_weight=1e3, style_weight=1e-2, learning_rate=5):

    #create images
    content = Image.load_image(content_path)
    style = Image.load_image(style_path)

    noise = Image.generate_noise_image()
    noise = tfe.Variable(noise, dtype=tf.float32)

    # create model
    vgg = CNN.VGG19()
    loss_weights = content_weight, style_weight
    layers_number = vgg.content_layers_num

    #create features
    content_features = vgg.get_content_features(content)
    style_features = vgg.get_style_features(style)
    gram_matrix_features = [Loss.g_matrix(feature) for feature in style_features]

    img_features = content_features, gram_matrix_features
    noise_features = vgg.get_output_feature(noise)

    #create optimizer
    opt = tf.train.AdamOptimizer(learning_rate, beta1=0.99, epsilon=1e-1)

    #store best results
    best_loss, best_img = float('inf'), None


    for i in range(iterations):

        grads, loss = Loss.compute_gradient(noise,noise_features,img_features,loss_weights,layers_number)

        opt.apply_gradients([(grads, noise)])

        clipped = Image.clip_image(noise)
        noise.assign(clipped)

        if loss < best_loss:

            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = Image.postprocess_image(noise.numpy())

        print(i)
        plot_img = noise.numpy()
        plot_img = Image.postprocess_image(plot_img)
        Image.show_image(plot_img)
        plt.show()

    return best_loss, best_img
