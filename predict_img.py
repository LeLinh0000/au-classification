import numpy as np
import os
import time
from keras.models import Model
from keras.layers import Dense
from keras.utils.np_utils import *
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from tensorflow.keras.optimizers import SGD
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import SGD
import json
def recognize_organs_fc_img():
    
    base_model = ResNet50(include_top=False, weights=None, pooling='avg')
    
    predictions = Dense(6, activation='softmax')(base_model.output)

    model = Model(inputs=base_model.input, outputs=predictions)
    weight_file = '/dataset/finetune/resnet50/finetune_weights_50_epoch.h5'
    assert os.path.exists(weight_file) is True, "Weight path is empty"
    model.load_weights(weight_file, by_name=False)

    # preprocess test data
    test_dir = '/dataset/img/input/'
    test_imgs = []
    test_imgs_original = []
    test_img_names = os.listdir(test_dir)
    start = time.process_time()
    for i in range(len(test_img_names)):
        img_path = os.path.join(test_dir, test_img_names[i])
        test_imgs_original.append(img_path)
        test_img = load_img(img_path)
        test_img = img_to_array(test_img)
        test_img = preprocess_input(test_img)
        # print(test_img)
        test_imgs.append(test_img)
        # test_class = test_img_names[i].split('-')[0]
        # test_labels.append(test_class)
    test_imgs = np.array(test_imgs)
    # encode the string label to integer
    organs = ['bladder', 'bowel', 'gallbladder', 'kidney', 'liver', 'spleen']
    mapping = {}
    for i in range(len(organs)):
        mapping[organs[i]] = i
    # for i in range(len(test_labels)):
    #     test_labels_int.append(mapping[test_labels[i]])

    # compile model
    learning_rate = 0.01
    decay_rate = 0
    momentum = 0.9
    sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    # predict with model
    test_logits = model.predict(test_imgs)
    test_predictions = np.argmax(test_logits, axis=1)
    end = time.process_time()
    total_time = end - start
    result = {}
    print("Average inference time for one image: {}".format(total_time/len(test_imgs)))
    for i in range(len(test_imgs)):
        print("predict: {}".format(organs[test_predictions[i]]))
        result = {'predict': organs[test_predictions[i]]}
    # result = {'label': 'n'}
    return (json.dumps(result))