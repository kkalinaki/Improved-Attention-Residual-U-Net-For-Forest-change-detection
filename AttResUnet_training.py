
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
from keras import backend, optimizers


import glob
path = "datasets"
os.chdir(path)

SIZE = 128 
n_classes = 4
image_dataset = []
name_of_images=[]
for directory_path in glob.glob("datasets/images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        name_of_images.append(img_path[19:-4])
        img = cv2.imread(img_path, 1)       
        img = cv2.resize(img, (SIZE, SIZE))
        image_dataset.append(img)
       
#Convert list to array for machine learning processing        
image_dataset = np.array(image_dataset)

#Capture mask/label info as a list
mask_dataset = [] 
for directory_path in glob.glob("datasets/masks/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        mask_dataset.append(mask)
        
#Convert list to array for machine learning processing          
mask_dataset = np.array(mask_dataset)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = mask_dataset.shape
mask_dataset_reshaped = mask_dataset.reshape(-1,1)
mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
mask_dataset_encoded_original_shape = mask_dataset_reshaped_encoded.reshape(n, h, w)

np.unique(mask_dataset_encoded_original_shape)

mask_dataset = np.expand_dims(mask_dataset_encoded_original_shape, axis=3)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.15, random_state = 0)

train_name_of_images=name_of_images[0:len(X_train)]
test_name_of_images=name_of_images[len(X_train):]

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X_train, y_train, test_size = 0.15, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 i

from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)

y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))


#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (128, 128, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (128, 128)), cmap='gray')
plt.show()

#######################################
#Parameters for model

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
n_classes = 4  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 16

#FOCAL LOSS AND DICE METRIC

from models import Attention_ResUNet, UNet, Attention_UNet, dice_coef, dice_coef_loss, jacard_coef, jacard_coef_loss, categorical_focal_loss

'''
UNet
'''
unet_model = UNet(input_shape)
unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss = categorical_focal_loss(gamma=2.0, alpha=0.25), 
             metrics=['accuracy', dice_coef])


print(unet_model.summary())

start1 = datetime.now() 
unet_history = unet_model.fit(X_train, y_train_cat, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test_cat ), 
                    shuffle=False,
                    epochs=100)

stop1 = datetime.now()
#Execution time of the model 
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)

unet_model.save('multiclass_unet_100epochs.hdf5')
#____________________________________________
'''
Attention UNet
'''
att_unet_model = Attention_UNet(input_shape)

att_unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss = categorical_focal_loss(gamma=2.0, alpha=0.25), 
             metrics=['accuracy', dice_coef])


print(att_unet_model.summary())
start2 = datetime.now() 
att_unet_history = att_unet_model.fit(X_train, y_train_cat, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test_cat ), 
                    shuffle=False,
                    epochs=100)
stop2 = datetime.now()
#Execution time of the model 
execution_time_Att_Unet = stop2-start2
print("Attention UNet execution time is: ", execution_time_Att_Unet)

att_unet_model.save('multiclass_Attention_UNet_100epochs.hdf5')

#___________________________________________
'''
Attention Residual Unet
'''
att_res_unet_model = Attention_ResUNet(input_shape)

att_res_unet_model.compile(optimizer=Adam(learning_rate = 1e-4), loss = categorical_focal_loss(gamma=2.0, alpha=0.25), 
             metrics=['accuracy', dice_coef])



print(att_res_unet_model.summary())


start3 = datetime.now() 
att_res_unet_history = att_res_unet_model.fit(X_train, y_train_cat, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test_cat ), 
                    shuffle=False,
                    epochs=100)
stop3 = datetime.now()

#Execution time of the model 
execution_time_AttResUnet = stop3-start3
print("Attention ResUnet execution time is: ", execution_time_AttResUnet)

att_res_unet_model.save('Multiclass_AttResUnet_100epochs.hdf5')

############################################################################
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting
import pandas as pd    
unet_history_df = pd.DataFrame(unet_history.history) 
att_unet_history_df = pd.DataFrame(att_unet_history.history) 
att_res_unet_history_df = pd.DataFrame(att_res_unet_history.history) 

with open('unet_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)
    
with open('att_unet_history_df.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)

with open('custom_code_att_res_unet_history_df.csv', mode='w') as f:
    att_res_unet_history_df.to_csv(f)    

#######################################################################
#Check history plots, one model at a time

#Unet history
history1 = unet_history

#plot the training and validation accuracy and loss at each epoch
loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
epochs = range(1, len(loss1) + 1)
plt.plot(epochs, loss1, 'y', label='Training loss')
plt.plot(epochs, val_loss1, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc1 = history1.history['dice_coef']
#acc = history.history['accuracy']
val_acc1 = history1.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc1, 'y', label='Training Dice')
plt.plot(epochs, val_acc1, 'r', label='Validation Dice')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()


#attention Unet history
history2 = att_unet_history

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
epochs = range(1, len(loss2) + 1)
plt.plot(epochs, loss2, 'y', label='Training loss')
plt.plot(epochs, val_loss2, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc2 = history2.history['dice_coef']
#acc = history.history['accuracy']
val_acc2 = history2.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc2, 'y', label='Training Dice')
plt.plot(epochs, val_acc2, 'r', label='Validation Dice')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()



#attention resnet unet history
history3 = att_res_unet_history

loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']
epochs = range(1, len(loss3) + 1)
plt.plot(epochs, loss3, 'y', label='Training loss')
plt.plot(epochs, val_loss3, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc3 = history3.history['dice_coef']
#acc = history.history['accuracy']
val_acc3 = history3.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc3, 'y', label='Training Dice')
plt.plot(epochs, val_acc3, 'r', label='Validation Dice')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()



#######################################################
#Model 1
#######################################################

model1 = unet_model
model_path1 = "multiclass_unet_100epochs.hdf5"
model = tf.keras.models.load_model(model_path1, compile=False)

# IoU and Prediction on testing Data for model 1
y_pred_do_not_use=model1.predict(X_do_not_use)
y_pred_do_not_use_argmax=np.argmax(y_pred_do_not_use, axis=3)

from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#plot some images for model 3 using testing data
import random
for i in range(len(X_test)):
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model1.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,:], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.savefig(r'datasets/%s'%test_name_of_images[i][26:]+'.png',dpi=300)
    plt.show()



# IoU and Prediction on Validation Data for model 1
y_pred=model1.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)


#plot some images for model 1
# import random
# test_img_number = random.randint(0, len(X_test)-1)
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img, 0)
# prediction = (model1.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='jet')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()


#######################################################
#Model 2
#######################################################
model2 = att_unet_model
model_path2 = "multiclass_Attention_UNet_100epochs.hdf5"
model = tf.keras.models.load_model(model_path2, compile=False)

# IoU and Prediction on testing Data for model 2
y_pred_do_not_use=model2.predict(X_do_not_use)
y_pred_do_not_use_argmax=np.argmax(y_pred_do_not_use, axis=3)

from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#plot some images for model 3 using testing data
for i in range(len(X_test)):
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model2.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,:], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.savefig(r'datasets/%s'%test_name_of_images[i][26:]+'.png',dpi=300)
    plt.show()

# IoU and Prediction on Validation Data for model 2
y_pred=model2.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)


#plot some images for model 2
# import random
# test_img_number = random.randint(0, len(X_test)-1)
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img, 0)
# prediction = (model2.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='jet')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()



#######################################################
#Model 3
#######################################################
model3 = att_res_unet_model
model_path3 = "Multiclass_AttResUnet_100epochs.hdf5"
model = tf.keras.models.load_model(model_path3, compile=False)

# IoU and Prediction on testing Data for model 3
y_pred_do_not_use=model3.predict(X_do_not_use)
y_pred_do_not_use_argmax=np.argmax(y_pred_do_not_use, axis=3)

from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_do_not_use[:,:,:,0], y_pred_do_not_use_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#plot some images for model 3 using testing data
for i in range(len(X_test)):
    test_img = X_do_not_use[i]
    ground_truth=y_do_not_use[i]
    #test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model3.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,:], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.savefig(r'datasets/%s'%test_name_of_images[i][26:]+'.png',dpi=300)
    plt.show()




#### IoU and Prediction on Validation Data for model 3 ###
y_pred=model3.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

#plot some images for model 3 using validation data
# import random
# test_img_number = random.randint(0, len(X_test)-1)
# test_img = X_test[test_img_number]
# ground_truth=y_test[test_img_number]
# #test_img_norm=test_img[:,:,0][:,:,None]
# test_img_input=np.expand_dims(test_img, 0)
# prediction = (model3.predict(test_img_input))
# predicted_img=np.argmax(prediction, axis=3)[0,:,:]


# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,0], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(ground_truth[:,:,0], cmap='jet')
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(predicted_img, cmap='jet')
# plt.show()
