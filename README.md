# GTSRB-Traffic Sign Recognition

![](https://github.com/Fcam34/GTSRB-CNN/blob/master/classes.jpg)

**Google.colab**

Google Colab is a free cloud service and now it supports free GPU.
*You can*
-improve your Python programming language coding skills.
-develop deep learning applications using popular libraries such as Keras, TensorFlow, PyTorch, and OpenCV.
The most important feature that distinguishes Colab from other free cloud services is; Colab provides GPU and is totally free.

```python
import zipfile
from google.colab import drive
import os

drive.mount('/content/drive/')

zip_ref = zipfile.ZipFile("/content/drive/My Drive/Colab Notebooks/Dataset/Dataset.zip", 'r')
zip_ref.extractall("/content/Dataset")
zip_ref.close()

if os.path.exists('/content/drive/My Drive/Colab Notebooks/Dataset/Training.npz'):
  !cp "/content/drive/My Drive/Colab Notebooks/Dataset/Training.npz" '/content/Dataset/Training.npz'
  !cp "/content/drive/My Drive/Colab Notebooks/Dataset/Test.npz" '/content/Dataset/Test.npz'

# copy the preprocessd dataset save in npz file from the colab to google drive
from google.colab import drive
drive.mount('/content/drive/')
!cp 'Dataset/Test.npz' "/content/drive/My Drive/Colab Notebooks/Dataset"
!cp 'Dataset/Training.npz' "/content/drive/My Drive/Colab Notebooks/Dataset"

# check the gpu information
# Tesla K80 is the best one in colab
!nvidia-smi  
```

**Dataset**

The problem we are gonna tackle is The German Traffic Sign Recognition Benchmark(GTSRB).
The dataset features 43 different signs under various sizes, lighting conditions, occlusions and is very similar to real-life data. Training set includes about 39000 images while test set has around 12000 images. Images are not guaranteed to be of fixed dimensions and the sign is not necessarily centered in each image. Each image contains about 10% border around the actual traffic sign.

Download ‘Images and annotations’ for training and test set from GTSRB website and extract them into a folder. Also download ‘Extended annotations including class ids’ file for test set. Organize these files so that directory structure looks like this:

![](https://github.com/Fcam34/GTSRB-CNN/blob/master/1587736298(1).jpg)

**Preprocessing** 

Histogram equalization
Standard image size
Get ROI target area

```python
import pandas as pd
from matplotlib import pyplot as plt

num_classes = 43
img_size = 48

path_training = r'Dataset/Final_Training/Images/'
path_test = r'Dataset/Final_Test/Images/'
path_test_csv = 'Dataset/GT-final_test.csv'
start_time = datetime.now()



def image_process(path, status_training, status_roi=True):
    global df
    # Get the Y value
    if status_training:
        label = int(path.split('/')[-2])
    else:
        label = int(df['ClassId'])

    img = PIL.Image.open(path)

    # plt.imshow(img)
    # plt.show()
    # Get the cropped image using Roi information
    if status_roi:
        box = (int(df['Roi.X1']), int(df['Roi.Y1']), int(df['Roi.X2']), int(df['Roi.Y2']))
        img = img.crop(box)

    # Implement the histogram equalization to get better image performance
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # Rescale the image to standard size
    img = transform.resize(img, (img_size, img_size))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    return img, label

# Load the training set
if os.path.exists('Dataset/Training.npz'):
    Training=np.load('Dataset/Training.npz')
    X = Training['X_train']
    Y = Training['Y_train']
else:
    imgs = []
    labels = []
    all_csv_paths = glob.glob(os.path.join(path_training, '*/*.csv'))  # Get all the csv file within the path

    for i, value in enumerate(all_csv_paths):
        all_csv_paths[i] = value.replace('\\', '/')

    for csv_path in all_csv_paths:  # Preprocess each image
        csv_file = pd.read_csv(csv_path, sep=';')
        for i in range(len(csv_file)):
            df = csv_file.loc[i]
            path = csv_path[:csv_path.rfind('/') + 1] + df['Filename']
            if os.path.exists(path):  # For Small training set, if the file exists
                img, label = (image_process(path, status_training=True))
                labels.append(label)
                imgs.append(img)

    X = np.array(imgs, dtype='float32')
    Y = np.eye(num_classes, dtype='uint8')[labels]  # Get one hot targets array
    if not os.path.exists('Dataset'):
        os.makedirs('Dataset')
    np.savez('Dataset/Training.npz',X_train=X,Y_train=Y)
    print(datetime.now() - start_time)

# Split the training set into training set and development set with the split ratio 80:20
X_train, X_dev, Y_train, Y_dev = train_test_split(X,Y,test_size=0.2, random_state=42)
print(X_train.shape)
print(X_dev.shape)
print(Y_train.shape)
print(Y_dev.shape)

# Load the test set
if os.path.exists('Dataset/Test.npz'):
    Test=np.load('Dataset/Test.npz')
    X_test = Test['X_test']
    Y_test = Test['Y_test']
else:
    labels = []
    imgs = []
    X_test = []
    Y_test = []
    test_file = pd.read_csv(path_test_csv, sep=';')
    for i in range(len(test_file)):
        df = test_file.loc[i]
        path = path_test + df['Filename']
        img, label = (image_process(path, status_training=False))
        imgs.append(img)
        labels.append(label)
    X_test = np.array(imgs, dtype='float32')
    Y_test = np.eye(num_classes, dtype='uint8')[labels]
    if not os.path.exists('Dataset'):
        os.makedirs('Dataset')
    np.savez('Dataset/Test.npz',X_test=X_test,Y_test=Y_test)
    print(datetime.now() - start_time)
```

**Model**

Let’s now define our models.

Go through the documentation of keras (relevant documentation : [here](https://keras.io/models/about-keras-models/)) to understand what parameters for each of the layers mean.

BPNN model

``` python
def bpnn_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(3,img_size,img_size)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
```


CNN model

```python
def cnn_model(parameter_list):
    model = Sequential()
    filter=parameter_list['Conv_Layer'][0]
    layer=parameter_list['Conv_Layer'][1]
    
    for i in range(layer):
        if i==0:
            if parameter_list['Conv_Overlap']:
                # filters, kernel size，padding，input data，activation function
                model.add(Conv2D(filter, (3, 3), padding='same',
                                 input_shape=(3, img_size, img_size),
                                 activation='relu'))
                model.add(Conv2D(filter, (3, 3), activation='relu'))
            else:
                model.add(Conv2D(filter, (3, 3), padding='same',
                                 input_shape=(3, img_size, img_size),
                                 activation='relu'))
        else:    
            if parameter_list['Conv_Overlap']:
                # filters, kernel size，padding，input data，activation function
                model.add(Conv2D(filter, (3, 3), padding='same',
                                 activation='relu'))
                model.add(Conv2D(filter, (3, 3), activation='relu'))
            else:
                model.add(Conv2D(filter, (3, 3), padding='same',
                                 activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if parameter_list['Dropout_Conv']!=-1:
            model.add(Dropout(parameter_list['Dropout_Conv']))
        filter=filter*2

    model.add(Flatten())

    units=parameter_list['Dense'][0]
    layer=parameter_list['Dense'][1]
    for i in range(layer):
        model.add(Dense(units))
        if parameter_list['BN']:
          model.add(BatchNormalization())
        model.add(Activation('relu'))
        if parameter_list['Dropout_Dense']!=-1:
            model.add(Dropout(parameter_list['Dropout_Dense']))

    model.add(Dense(num_classes, activation='softmax'))
    return model

```


**Train**

Now, our model is ready to train. During the training, our model will iterate over batches of training set, each of size batch_size. For each batch, gradients will be computed and updates will be made to the weights of the network automatically. One iteration over all the training set is referred to as an epoch. Training is usually run until the loss converges to a constant.

We will add a couple of features to our training:

Learning rate scheduler : Decaying learning rate over the epochs usually helps model learn better
Model checkpoint : We will save the model with best validation accuracy. This is useful because our network might start overfitting after a certain number of epochs, but we want the best model.
These are not necessary but they improve the model accuracy. These features are implemented via callback feature of Keras. callback are a set of functions that will applied at given stages of training procedure like end of an epoch of training. Keras provides inbuilt functions for both learning rate scheduling and model checkpointing.

```python
# Calculate the learning rate during each epoch
def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

# Train the model using SGD + momentum
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

# Function to compile the model
def model_compile():
    global model
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    
# Function to train the model
def model_train(batch_size,epoch,test=False):
    global model
    if test:
      validation_set=(X_test,Y_test)
    else:
      validation_set=(X_dev,Y_dev)     
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        validation_data=validation_set,
                        shuffle=True,
                        callbacks=[LearningRateScheduler(lr_schedule),
                                   # Save the best epoch report into model.h5
                                   ModelCheckpoint('Model.h5', save_best_only=True)]
                        # h5 is the format for keras to save the model
                        )
    return history

def plot_confusion_matrix(con_matrix): #Visualization of confusion matrix
    plt.imshow(con_matrix)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    labels = []
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# Function to clear the model after using
def model_clear():
  global model
  #K.clear_session()
  #tf.reset_default_graph()
  del model

# Function to predict the model
# Get the performance on the test set
def model_predict(test=False):
    global model
    if test:
        Y_pred = np.eye(num_classes, dtype='uint8')[model.predict_classes(X_test)]
        acc = np.sum(Y_pred == Y_test) / np.size(Y_pred)

        print("Accuracy on test set: {}".format(acc))
        
        con_matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))

        
        FP = con_matrix.sum(axis=0) - np.diag(con_matrix)  
        FN = con_matrix.sum(axis=1) - np.diag(con_matrix)
        TP = np.diag(con_matrix)
        TN = con_matrix.sum() - (FP + FN + TP)
        print(FP,FN,TP,TN)


        
        print("confusion_matrix on test set: {}".format(con_matrix))
        plot_confusion_matrix(con_matrix)
        precision = precision_score(Y_test.argmax(axis=1), Y_pred.argmax(axis=1),average='macro')
        print("precision on test set: {}".format(precision))
        recall = recall_score(Y_test.argmax(axis=1), Y_pred.argmax(axis=1),average='macro')
        f1=f1_score(Y_test.argmax(axis=1), Y_pred.argmax(axis=1),average='macro')
        
        print("recall on test set: {}".format(recall))
        print("f1 on test set: {}".format(f1))
        
        #con_mat=confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
        
        #con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]    
        
        #print(con_mat_norm) #confusion matrix
    else:
        Y_pred = np.eye(num_classes, dtype='uint8')[model.predict_classes(X_dev)]
        acc = np.sum(Y_pred == Y_dev) / np.size(Y_pred)
        print("Accuracy on development set: {}".format(acc))

# The whole process of training the model
def model_process(batch_size,epoch,test=False):
    global model
    model_compile()
    model_train(batch_size=batch_size,epoch=epoch,test=test)
    model_predict(test)
    model_clear()


# Train the bp neutral network model
print('BPNN:')
model = bpnn_model()
model_process(batch_size=32,epoch=5)

batch_size=32
epoch=5

# Try the CNN model without dropout layer
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':False,
                'Dropout_Conv':-1,'Dense':(512,1),'Dropout_Dense':-1,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with dropout layer
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':False,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with deeper convolutional layer
parameter_list={'Conv_Layer':(32,5),'Conv_Overlap':False,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with deeper hidden layer
parameter_list={'Conv_Layer':(32,5),'Conv_Overlap':False,
                'Dropout_Conv':0.2,'Dense':(512,3),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with Two convolutional layers together
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with different dropout probability
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.5,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with different dropout probability
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.2,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with batch normalisation
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':True}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

# Try the CNN model with different batch sizes
batch_size=16
epoch=5
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
print('Batch size=',batch_size)
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

batch_size=64
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
print('Batch size=',batch_size)
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)

batch_size=256
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
print('Batch size=',batch_size)
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch)
```

You will see:

![](https://github.com/Fcam34/GTSRB-CNN/blob/master/1587735924(1).jpg)

The BPNN model has been implemented with four dense layers with the dimension of 512
along with the batch normalisation after each layer. The model achieved an accuracy of
99.896%. We also check the performance of the CNN model with the accuracy of 99.964%
which was slightly higher than the BPNN model.

As the CNN model achieved higher accuracy than BPNN model. For the next step, we’d
like to tune the parameters of the CNN model to check what permutation of different
parameters can achieve the best result.
The final parameter choice of the best model would be
{'Batch_Size': 32, 'Conv_Layer': (32, 3), 'Conv_Overlap': True, 'Dropout_Conv': 0.2, 'Dense': (512, 1), 'Dropout_Dense': 0.5, 'BN': False} 

**Evaluation**

Let’s quickly load test data and evaluate our model on it:

```python
X_train=X
Y_train=Y
batch_size=32
epoch=30
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process(batch_size=batch_size,epoch=epoch,test=True) 
```
Accuracy on test set: 0.9992450606713436

precision on test set: 0.9804020331235772

recall on test set: 0.9768294996861732

f1 on test set: 0.9774262802846536

**Data Augmentation**

Our model has 1358155 parameters (try model.count_params() or model.summary()). That’s 4X the number of training images.

If we can generate new images for training from the existing images, that will be a great way to increase the size of the dataset. This can be done by slightly

-translating of image
-rotating of image
-Shearing the image
-Zooming in/out of the image

Rather than generating and saving such images to hard disk, we will generate them on the fly during training. This can be done directly using built-in functionality of keras.

```python
from keras.preprocessing.image import ImageDataGenerator

#Data Augmentation
datagen = ImageDataGenerator(featurewise_center=False, 
                            featurewise_std_normalization=False, 
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.2,
                            horizontal_flip=True,
                            rotation_range=10.,)

datagen.fit(X_train)

#after Data Augmentation
def model_train2(batch_size,epoch):
    global model
    history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
                                  epochs=epoch,
                                  steps_per_epoch=0.2,
                                  validation_data=(X_test,Y_test),
                                  shuffle=True,
                                  callbacks=[LearningRateScheduler(lr_schedule),
                                             ModelCheckpoint('Model.h5', save_best_only=True)]
                                  )
    return history

def model_process2(batch_size,epoch,test=False):
    global model
    model_compile()
    model_train2(batch_size=batch_size,epoch=epoch)
    model_predict(test)
    model_clear()

# Get the performance on test set using 30 epoches and the selected best model
# Set the training set back to full
X_train=X
Y_train=Y
batch_size=32
epoch=30
parameter_list={'Conv_Layer':(32,3),'Conv_Overlap':True,
                'Dropout_Conv':0.2,'Dense':(512,1),'Dropout_Dense':0.5,'BN':False}
print('CNN: '+str(parameter_list))
model=cnn_model(parameter_list)
model_process2(batch_size=batch_size,epoch=epoch,test=True)  

```
Accuracy on test set: 0.9561325010587564

With this model, We get 95.61% accuracy on test set.
