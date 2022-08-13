import os
import tensorflow as tf
from Helper_functions import pred_and_plot

if __name__ == '__main__':
    base_path = r'C:\Users\vikassaigiridhar\Music\food_101_test\Test_Images'
    pred_image = r"C:\Users\vikassaigiridhar\Music\food_101_test\f101_test\376861.jpg"
    model = tf.keras.models.load_model('./Models')
    class_names = [name for name in os.listdir(base_path)]
    #print(class_names)
    pred_class,pred_prob = pred_and_plot(model,pred_image,class_names)
    print(f'Predicted class is {pred_class} with a probability of {pred_prob}')