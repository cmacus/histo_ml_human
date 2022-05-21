#!/usr/bin/env python
# coding: utf-8

# # Histology image classification

# ## Import necessary libraries

# In[ ]:


get_ipython().system('pip install visualkeras')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle, cv2, os
import visualkeras
from sklearn.metrics import confusion_matrix
import pandas as pd


# In[ ]:


plt.style.use('seaborn')


# ## Define tissue classes

# In[ ]:


classes = ['cerebellum','cerebrum','heart','kidney','liver','lung', 'pancreas','skin','stomach','trachea']


# ## Load images from train folder

# In[ ]:


IMAGE_DIMS = (128,96,3)

# Build model
data = []
labels = []
train_dir = "data_512/train"

imagePaths = sorted(list(paths.list_images(train_dir)))

# loop over the input images
for imagePath in imagePaths:
  foldername = os.path.dirname(imagePath).split('/')[-1]
  # load the image, pre-process it, and store it in the data list
  image = cv2.imread(imagePath)
  image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
  image = img_to_array(image)
  data.append(image)
	# extract set of class labels from the image path and update the
	# labels list
  labels.append(foldername)

#normalise
data = np.array(data, dtype="float32") / 255.0

labels = np.array(labels)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# ## Build VGG 16 model (VML)

# In[ ]:


INIT_LR = 1e-3
NUM_EPOCHS = 30
BATCH_SIZE = 10
losses = {"class_label": "categorical_crossentropy"}

# define a dictionary that specifies the weights per loss
lossWeights = {"class_label": 1.0,}


# In[ ]:


vgg = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(128, 96, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax",name="class_label")(softmaxHead)

model = Model(inputs=vgg.input,outputs=softmaxHead)



# initialize the optimizer, compile the model, and show the model summary
opt = Adam(lr=INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())


# ## Train model

# In[ ]:


# Train
H = model.fit(
	trainX, trainY,
	validation_data=(testX, testY),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
model.save('vml.model', save_format="h5")
# serialize the label binarizer to disk
f = open('lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


# In[ ]:


# plot the total loss
items = ["loss","accuracy"]
N = np.arange(0, NUM_EPOCHS)
plt.figure(figsize=(10,5))
plt.style.use("ggplot")
plt.plot(N,H.history['loss'],label='loss')
plt.plot(N,H.history['accuracy'],label='accuracy')
plt.xticks(N)
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.close()


# In[ ]:


# Show individual images

def predict_image(impath):
 image = cv2.imread(impath)
 
 # pre-process the image for classification
 image = cv2.resize(image, (96, 128))
 image = image.astype("float") / 255.0
 image = img_to_array(image)
 image = np.expand_dims(image, axis=0)

 # load the trained convolutional neural network and the multi-label
 # binarizer
 print("Loading...")
 model = load_model('vml.model')
 lb = pickle.loads(open('lb.pickle', "rb").read())
 # classify the input image then find the indexes of the two class
 # labels with the *largest* probability
 print("Classifying image...")
 proba = model.predict(image)[0]
 idxs = np.argsort(proba)[::-1][:2] # Top two

 # show the probabilities for each of the individual labels
 for (label, p) in zip(lb.classes_, proba):
   print("{}: {:.2f}%".format(label, p * 100))

 plt.figure(figsize=(20,5))
 plt.subplot(121)
 plt.grid(b=None)
 plt.imshow(plt.imread(impath))
 plt.subplot(122)
 plt.grid(b=None)
 plt.bar(lb.classes_,proba)
 plt.xticks(rotation=45)


# In[ ]:


impath = "data_512/train/skin/210622162613.png"
predict_image(impath)


# In[ ]:


impath = "data_512/val/kidney/210621160944.png"
predict_image(impath)


# In[ ]:


# Run validation images

val_dir = "data_512/val"
lst_fnames = []
model = load_model('vml.model')
lb = pickle.loads(open('lb.pickle', "rb").read())
for impath in (sorted(list(paths.list_images(val_dir)))):
  image = cv2.imread(impath)
  # pre-process the image for classification
  image = cv2.resize(image, (96, 128))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  original = os.path.dirname(impath).split('/')[-1]
  proba = model.predict(image)[0] 
  idxs = np.argsort(proba)[::-1][:1] # Top 
  predicted = classes[idxs[0]]
  lst_fnames.append([os.path.basename(impath),original, predicted])


# In[ ]:


# Save predictions with filenames
pickle.dump(lst_fnames,open('label_filenames.pickle','wb'))


# In[ ]:


lst = [[i[1],i[2]] for i in lst_fnames]


# In[ ]:


# Save predictions
pickle.dump(lst,open('preds.pickle','wb'))


# In[ ]:


# Load predictions
preds = pickle.load(open('preds.pickle','rb'))
len(preds)


# In[ ]:



import pandas as pd
lst = np.array(lst)
y_actu = pd.Series(lst[:,0], name='Actual')
y_pred = pd.Series(lst[:,1], name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'])
df_confusion


# In[ ]:


confusion_matrix(y_actu, y_pred)


# In[ ]:


pd.crosstab(y_actu,y_pred)


# In[ ]:


import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.figure(figsize=(20,10))
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
plot_confusion_matrix(df_confusion)


# ## Set 1 quiz images

# In[1]:


## Set 1 quiz images
set_1 = ['210622153809.png',
 '210621162752.png',
 '210621161033.png',
 '210621154132.png',
 '210621155552.png',
 '210621162747.png',
 '210622154520.png',
 '210622160746.png',
 '210621153410.png',
 '210621162119.png',
 '210621161857.png',
 '210621155545.png',
 '210621154552.png',
 '210621154628.png',
 '210621160612.png',
 '210621154951.png',
 '210621153449.png',
 '210621154649.png',
 '210621162358.png',
 '210622152712.png',
 '210621153407.png',
 '210621161021.png',
 '210621153350.png',
 '210622153817.png',
 '210621145300.png',
 '210621160840.png',
 '210622153518.png',
 '210621153041.png',
 '210621155611.png',
 '210622155249.png',
 '210621152821.png',
 '210621162708.png',
 '210621153030.png',
 '210621145232.png',
 '210621155529.png',
 '210621162639.png',
 '210622160607.png',
 '210621152118.png',
 '210622162138.png',
 '210622155223.png',
 '210622160712.png',
 '210621160930.png',
 '210621154733.png',
 '210621154054.png',
 '210621153635.png',
 '210622160827.png',
 '210622155608.png',
 '210621153936.png',
 '210621154729.png',
 '210621152946.png',
 '210622152714.png',
 '210622153718.png',
 '210621153842.png',
 '210622154914.png',
 '210621153459.png',
 '210621153042.png',
 '210621161524.png',
 '210621145338.png',
 '210621153447.png',
 '210622161416.png',
 '210621161113.png',
 '210621151423.png',
 '210622153722.png',
 '210621161920.png',
 '210621153637.png',
 '210621161034.png',
 '210621151314.png',
 '210621153452.png',
 '210621162208.png',
 '210621145426.png',
 '210621155312.png',
 '210621162732.png',
 '210621155450.png',
 '210622155602.png',
 '210622162512.png',
 '210621161025.png',
 '210621162717.png',
 '210621155335.png',
 '210621162816.png',
 '210621154650.png',
 '210621154104.png',
 '210621160631.png',
 '210622153504.png',
 '210621154852.png',
 '210621161527.png',
 '210621153514.png',
 '210621154820.png',
 '210622162515.png',
 '210621155116.png',
 '210621155456.png',
 '210621145235.png',
 '210621152740.png',
 '210622154814.png',
 '210621151420.png',
 '210621153837.png',
 '210621145130.png',
 '210622155251.png',
 '210621145745.png',
 '210621160436.png',
 '210622153801.png']


# In[ ]:


val_dir = "data_512/val"
lst_fnames = []
model = load_model('vml.model')
lb = pickle.loads(open('lb.pickle', "rb").read())
for impath in (sorted(list(paths.list_images(val_dir)))):
  if (os.path.basename(impath) in set_1):
    image = cv2.imread(impath)
    # pre-process the image for classification
    image = cv2.resize(image, (96, 128))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    original = os.path.dirname(impath).split('/')[-1]
    proba = model.predict(image)[0] 
    idxs = np.argsort(proba)[::-1][:1] # Top 
    predicted = classes[idxs[0]]
    lst_fnames.append([os.path.basename(impath),original, predicted,proba])


# In[ ]:


proba


# In[ ]:


pickle.dump(lst_fnames,open('set_1_preds_VML.pickle','wb'))


# ### Errors by VML model in set 1 quiz images

# In[ ]:


get_ipython().run_line_magic('cd', "'data_512/val'")


# In[ ]:


confusing_images_set_1 = [('kidney', '210621162358.png'),
 ('stomach', '210621145300.png'),
 ('kidney', '210621160930.png'),
 ('stomach', '210621145338.png'),
 ('kidney', '210621161034.png'),
 ('stomach', '210621145426.png'),
 ('stomach', '210622162515.png'),
 ('trachea', '210622154814.png'),
 ('skin', '210621160436.png')]


# In[ ]:


plt.style.use('seaborn')
for c in confusing_images_set_1:
  predict_image('data_512/val/' + c[0] + '/' + c[1])


# In[ ]:


# Visualise layers
# Show individual images

def visualise_layers(impath):
  plt.jet()
  image = cv2.imread(impath)
  # pre-process the image for classification
  image = cv2.resize(image, (96, 128))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  # load the trained convolutional neural network and the multi-label
  # binarizer
  print("[INFO] loading network...")
  model = load_model('vml.model')
  lb = pickle.loads(open('lb.pickle', "rb").read())
  layer_names = [layer.name for layer in model.layers]
  layer_outputs = [layer.output for layer in model.layers]
  feature_map_model = Model(inputs=model.input, outputs=layer_outputs)

  feature_maps = feature_map_model.predict(image)
  for layer_name, feature_map in zip(layer_names, feature_maps):
    if ('conv' in layer_name) and (len(feature_map.shape) == 4): # Convolution layers only
      k = feature_map.shape[-1]  
      size=feature_map.shape[1]
      for i in range(k):
        feature_image = feature_map[0, :, :, i]
        feature_image-= feature_image.mean()
        feature_image/= feature_image.std ()
        feature_image*=  64
        feature_image+= 128
        if i == 0:
          image_belt = feature_image
        else:
          image_belt = np.hstack((image_belt, feature_image))
      s = image_belt.shape
      plt.figure(figsize=(20,5))
      plt.grid(False)
      plt.imshow(image_belt[:,:1000])
      plt.show()


# In[ ]:


# This trachea was misclassified as cerebellum
visualise_layers('data_512/val/trachea/210622154814.png')


# In[ ]:


# Compare with an actual cerebellum
visualise_layers('data_512/val/cerebellum/210622153722.png')


# In[ ]:


visualise_layers('data_512/val/liver/210621154552.png')


# In[ ]:


for f in os.listdir('data_512/val/stomach'):
  print(f)
  predict_image('data_512/val/stomach/%s' % f)


# In[ ]:


visualise_layers('data_512/val/stomach/210621145426.png')


# In[ ]:


## Draw model
model = load_model('vml.model')
visualkeras.layered_view(model)


# ## Set 2 (OTS) images

# In[ ]:


set_2_folder = 'data_512/set_2'
lst_fnames_set_2 = []
model = load_model('vml.model')
lb = pickle.loads(open('lb.pickle', "rb").read())
for impath in (sorted(list(paths.list_images(set_2_folder)))):
  image = cv2.imread(impath)
  # pre-process the image for classification
  image = cv2.resize(image, (96, 128))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  original = os.path.dirname(impath).split('/')[-1]
  proba = model.predict(image)[0] 
  idxs = np.argsort(proba)[::-1][:1] # Top 
  predicted = lb.classes_[idxs[0]]
  lst_fnames_set_2.append([os.path.basename(impath),original, predicted])


# In[ ]:


lst_fnames_set_2


# ## Build Resnet50 model

# In[ ]:


resnet_model = Sequential()
pretrained_model= ResNet50(include_top=False,
                   input_shape=(128,96,3),
                   pooling='avg',classes=10,
                   weights='imagenet')
resnet_model.add(pretrained_model)


# In[ ]:


resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation="relu"))
resnet_model.add(Dropout(0.5))
resnet_model.add(Dense(512, activation="relu"))
resnet_model.add(Dropout(0.5))
resnet_model.add(Dense(len(lb.classes_), activation="softmax",name="class_label"))


# In[ ]:


resnet_model.summary()


# In[ ]:


resnet_model.compile(optimizer=Adam(lr=INIT_LR),loss=losses,metrics=['accuracy'])


# In[ ]:


history = resnet_model.fit(x=trainX,y=trainY,batch_size=BATCH_SIZE,validation_data=(testX,testY), epochs=NUM_EPOCHS)


# ## Build an InceptionV2 (IML) model

# In[ ]:


inception_model = Sequential()
pretrained_model= InceptionResNetV2(include_top=False,
                   input_shape=(128,96,3),
                   pooling='avg',classes=10,
                   weights='imagenet')
for layer in pretrained_model.layers[:100]:
  layer.trainable= False
inception_model.add(pretrained_model)


# In[ ]:


inception_model.add(Flatten())
inception_model.add(Dense(512, activation="relu"))
inception_model.add(Dropout(0.5))
inception_model.add(Dense(512, activation="relu"))
inception_model.add(Dropout(0.5))
inception_model.add(Dense(len(lb.classes_), activation="softmax",name="class_label"))


# In[ ]:


inception_model.summary()


# In[ ]:


inception_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


history = inception_model.fit(x=trainX,y=trainY,batch_size=10,validation_data=(testX,testY), epochs=30)


# In[ ]:


plt.plot(history.history['accuracy'],label='Accuracy')
plt.plot(history.history['loss'],label='Loss')
plt.legend()


# In[ ]:


# Run validation images

val_dir = "data_512/val"
lst_fnames = []
model = load_model('inception.model')
lb = pickle.loads(open('lb.pickle', "rb").read())
for impath in (sorted(list(paths.list_images(val_dir)))):
  image = cv2.imread(impath)
  # pre-process the image for classification
  image = cv2.resize(image, (96, 128))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  original = os.path.dirname(impath).split('/')[-1]
  proba = model.predict(image)[0] 
  idxs = np.argsort(proba)[::-1][:1] # Top 
  predicted = classes[idxs[0]]
  lst_fnames.append([os.path.basename(impath),original, predicted])


# In[ ]:


pickle.dump(lst_fnames,open('val_preds_IML.pickle','wb'))


# In[ ]:


lst = [[i[1],i[2]] for i in lst_fnames]


# In[ ]:


import pandas as pd
lst = np.array(lst)
y_actu = pd.Series(lst[:,0], name='Actual')
y_pred = pd.Series(lst[:,1], name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'])
df_confusion


# In[ ]:


visualkeras.layered_view(inception_model)


# In[ ]:


inception_model.save("inception.model")


# In[ ]:


# Show individual images

def predict_image_inception(impath):
 image = cv2.imread(impath)
 
 # pre-process the image for classification
 image = cv2.resize(image, (96, 128))
 image = image.astype("float") / 255.0
 image = img_to_array(image)
 image = np.expand_dims(image, axis=0)

 # load the trained convolutional neural network and the multi-label
 # binarizer
 print("Loading...")
 model = load_model('inception.model')
 lb = pickle.loads(open('lb.pickle', "rb").read())
 # classify the input image then find the indexes of the two class
 # labels with the *largest* probability
 print("Classifying image...")
 proba = model.predict(image)[0]
 idxs = np.argsort(proba)[::-1][:2] # Top two

 # show the probabilities for each of the individual labels
 for (label, p) in zip(lb.classes_, proba):
   print("{}: {:.2f}%".format(label, p * 100))

 plt.figure(figsize=(20,5))
 plt.subplot(121)
 plt.grid(b=None)
 plt.imshow(plt.imread(impath))
 plt.subplot(122)
 plt.grid(b=None)
 plt.bar(lb.classes_,proba)
 plt.xticks(rotation=45)


# In[ ]:


impath = "data_512/train/skin/210622162613.png"
predict_image_inception(impath)


# In[ ]:


impath = "data_512/val/kidney/210621160944.png"
predict_image_inception(impath)


# In[ ]:


val_dir = "data_512/val"
lst_fnames = []
model = load_model('inception.model')
lb = pickle.loads(open('lb.pickle', "rb").read())
for impath in (sorted(list(paths.list_images(val_dir)))):
  if (os.path.basename(impath) in set_1):
    image = cv2.imread(impath)
    # pre-process the image for classification
    image = cv2.resize(image, (96, 128))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    original = os.path.dirname(impath).split('/')[-1]
    proba = model.predict(image)[0] 
    idxs = np.argsort(proba)[::-1][:1] # Top 
    predicted = classes[idxs[0]]
    lst_fnames.append([os.path.basename(impath),original, predicted,proba])


# In[ ]:


pickle.dump(lst_fnames,open("set_1_preds_IML.pickle","wb"))


# In[ ]:


lst = [[i[1],i[2]] for i in lst_fnames]


# In[ ]:


lst = np.array(lst)
y_actu = pd.Series(lst[:,0], name='Actual')
y_pred = pd.Series(lst[:,1], name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred,rownames=['Actual'], colnames=['Predicted'])
df_confusion


# In[ ]:


set_2_folder = 'data_512/set_2'
lst_fnames_set_2 = []
model = load_model('inception.model')
lb = pickle.loads(open('lb.pickle', "rb").read())
for impath in (sorted(list(paths.list_images(set_2_folder)))):
  image = cv2.imread(impath)
  # pre-process the image for classification
  image = cv2.resize(image, (96, 128))
  image = image.astype("float") / 255.0
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  original = os.path.dirname(impath).split('/')[-1]
  proba = model.predict(image)[0] 
  idxs = np.argsort(proba)[::-1][:1] # Top 
  predicted = lb.classes_[idxs[0]]
  lst_fnames_set_2.append([os.path.basename(impath),original, predicted])


# In[ ]:


lst = [[i[0],i[1],i[2]] for i in lst_fnames_set_2]


# In[ ]:


lst

