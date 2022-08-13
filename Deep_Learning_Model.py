import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Conv2D
from tensorflow.keras.layers import MaxPool2D,BatchNormalization,GlobalAvgPool2D,Dense,GlobalAveragePooling2D

data_augumentation = Sequential([
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2),
        preprocessing.RandomZoom(0.3),
        preprocessing.Rescaling(1 / 255.)
    ], name='data_augumentation')

#building the model from scratch
def food_classification_model(no_of_classes):
    my_input = Input(shape = (400,400,3))

    x = Conv2D(32,(3,3),activation = 'relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(128,activation = 'relu')(x)
    x = Dense(no_of_classes,activation = 'softmax')(x)

    return Model(inputs = my_input,outputs = x)

#using a pretrained model [feature extraction]
def food_classification_model_pretrained(url,no_of_classes):
    feature_extractor_layer = hub.KerasLayer(url,
                                             trainable=False, #freezing the learned patterns )
                                             name = 'feature_layer_extraction',
                                             input_shape = (227,227,3))
    #creating the model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(no_of_classes,activation = 'softmax',name = 'output_layer')
    ])

    return model

#including the augumentation layer for efficient GPU usage for model built from scratch
def food_101_with_augumented_layer_from_scratch(no_of_classes):

    my_input = Input(shape=(227,227,3))

    x = data_augumentation(my_input) #happens only during training

    x = Conv2D(32,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(no_of_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

#including the augumentation layer in the model for a pre_trained model
def Food_101_pretrained_with_aug_layer(base_model,no_of_classes):

    Input_layer = layers.Input(shape=(227,227,3),name='Input_Layer') #shape of input layer
    x = data_augumentation(Input_layer)
    x = base_model(x,training = False)
    x = layers.GlobalAvgPool2D()(x)
    Output_layer = layers.Dense(no_of_classes,activation='softmax',name='output_layer')(x)

    return Model(inputs = Input_layer,outputs = Output_layer)



if __name__ == '__main__':
    n = 'model_url'
    model = food_classification_model_pretrained(n, 5)
    #model = food_classification_model(5)
    model.summary()