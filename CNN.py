import tensorflow.keras as models

from tensorflow import enable_eager_execution
from tensorflow.keras.applications.vgg19 import VGG19

class VGG19_c:

    def __init__(self):

        # here you say where you want to take the features for the content
        self.contentLayers = ['block4_conv2']

        # here you say where you want to take the features for the style
        self.styleLayers = ['block1_conv1',
                              'block2_conv1',
                              'block3_conv1',
                              'block4_conv1',
                              'block5_conv1']
        self.content_layers_num = len(self.contentLayers)
        self.style_layers_num = len(self.styleLayers)


        self.model = self.getModel()

        # after setting model not trainable we also set the layers not trainable
        for layer in self.model.layers:
            layer.trainable = False


    def get_content_features(self,img):
        return self.get_output_features(img)[0]


    def get_style_features(self, img):
        return self.get_output_features(img)[1]


    def get_output_features(self, content):

        features = self.model(content)

        # for the content take only the content layers from 0 to len of content
        content = [style_content[0] for style_content in features[self.style_layers_num:]]

        # for style take only the style layers from len of content to len of content + len of style
        style = [style[0] for style in features[:self.style_layers_num]]

        return content, style


    def getModel(self):

        # we load the VGG19 pretrained with the dataset imagenet and we don't include the 3 fully connected layers on
        # top of theVGG19
        vgg = VGG19(include_top=False, weights='imagenet')

        # we freeze the weights and the variables
        vgg.trainable = False

        style_feature = []
        for i in self.styleLayers:
            style_feature.append(vgg.get_layer(i).output)

        content_feature = []
        for i in self.contentLayers:
            content_feature.append(vgg.get_layer(i).output)

        #using the Keras API we return the model of the CNN
        return models.Model(vgg.input, style_feature + content_feature)










