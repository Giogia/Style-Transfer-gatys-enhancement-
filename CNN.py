import tensorflow as tf
import tensorflow.python.keras as models
class VGG19:
    def __init__(self, content, style):
        tf.enable_eager_execution() #you enable eager eecution because we want the fow to be linear
        self.content = content
        self.style = style
        self.styleLayers = ['block5_conv2']       #here you say where you want to take the features for the style
        self.contentLayers = ['block1_conv1',
                              'block2_conv1',     #here you say where you want to take the features for the content
                              'block3_conv1',
                              'block4_conv1',
                              'block5_conv1']
        self.model = self.getModel()



    def getModel(self):

        # we load the VGG19 pretrained with the dataset imagenet and we don't include the 3 fully connected layers on
        # top of theVGG19

        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

        # we freeze the weights and the variables

        vgg.trainable = False#we load the VGG19 pretrained with the dataset imagenet and we don't include the 3 fully connected layers on
        #top of theVGG19

        vgg = tf.keras.applications.vgg19.VGG19(include_top=False,   weights='imagenet')

        #we freeze the weights and the variables

        vgg.trainable = False

        style_feature = []

        for i in self.styleLayers:
            style_feature.append(self.vgg.get_layer(i).output)
        content_feature = []
        for i in self.contentLayers:
            content_feature.append(self.vgg.get_layer(i).output)

        #using the Keras API we return the model of the CNN

        return models.Model(self.vgg.input, style_feature + content_feature)










