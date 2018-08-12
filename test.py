# show images inline
#matplotlib inline

# automatically reload modules when they have changed
#load_ext autoreload
#autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('.', 'snapshots', 'model.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
#print(model.summary())
# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.load_model(model_path, backbone_name='resnet50', convert_model=True)
#model.summary()
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}
# load image
image_paths = []
#input_path="D:/postgraduateworking/dataset/KITTI/testing/image_2/"
input_path="D:/postgraduateworking/dataset/KITTI/rain/"
if os.path.isdir(input_path):
    for inp_file in os.listdir(input_path):
        image_paths += [input_path + inp_file]
fileObject1=open('time1/time.txt' , 'w')
for image_path in image_paths:
    (path1,file1)=os.path.split(image_path)
    (filename1,filename2)=os.path.splitext(file1)
    fileObject = open('txtresult1/%s.txt' % filename1, 'w')

    image = read_image_bgr(image_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    plt.imshow(draw)
    plt.show()
    #process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    fps=1/(time.time()-start)
    fileObject1.write(str(fps))
    fileObject1.write('\t')
    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        score1='{:.3f}'.format(score)
        fileObject.write(labels_to_names[label])
        fileObject.write('\t')
        fileObject.write('-1')
        fileObject.write('\t')
        fileObject.write('-1')
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write(str(b[0]))
        fileObject.write('\t')
        fileObject.write(str(b[1]))
        fileObject.write('\t')
        fileObject.write(str(b[2]))
        fileObject.write('\t')
        fileObject.write(str(b[3]))
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write('0.0')
        fileObject.write('\t')
        fileObject.write(str(score1))
        fileObject.write('\t')
        fileObject.write('\n')
        draw_caption(draw, b, caption)

    plt.figure()
    plt.axis('off')
    #plt.savefig('%s'%file1)
    plt.imshow(draw)
    #plt.show()
    fileObject.close()
    plt.savefig("outimage/%s.png" % file1)
    plt.show()
    #cv2.imwrite('./images/' + image_path.split('/')[-1], np.uint8(draw))
fileObject1.close()