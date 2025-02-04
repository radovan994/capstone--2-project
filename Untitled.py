#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import seaborn as sns

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#EDA


# In[3]:


base_dir = './capstone-project'


# In[4]:


train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# In[5]:


def num_of_classes(folder_dir, folder_name) :
    classes = [class_name for class_name in os.listdir(train_dir)]
    print(f'number of classes in {folder_name} folder : {len(classes)}')


# In[6]:


num_of_classes(train_dir, 'train')
num_of_classes(validation_dir, 'validation')
num_of_classes(test_dir, 'test')


# In[7]:


classes = [class_name for class_name in os.listdir(train_dir)]
count = []
for class_name in classes :
    count.append(len(os.listdir(os.path.join(train_dir, class_name))))

plt.figure(figsize=(15, 4))
ax = sns.barplot(x=classes, y=count, color='navy')
plt.xticks(rotation=285)
for i in ax.containers:
    ax.bar_label(i,)
plt.title('Number of samples per label', fontsize=25, fontweight='bold')
plt.xlabel('Labels', fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.yticks(np.arange(0, 105, 10))
plt.show()


# In[8]:


#balanced set


# In[9]:


#getting glimpse into dataset
all_classes = os.listdir(base_dir+ "/train/")


# In[10]:


import random
random_img = random.choice(os.listdir(base_dir+ "/train/"+ all_classes[0]))
random_img_path = os.path.join(base_dir+ "/train/" + all_classes[0] + "//" + random_img)
load_img(random_img_path, target_size=(128,128))


# In[11]:


all_classes


# In[12]:


fig, axs = plt.subplots (nrows = 8, ncols = 4, figsize=(16,24))

class_num = 0 
for XX, ax in enumerate(fig.axes): #32
    class_num = ( 0 if class_num>7 else class_num ) #set a condition so it wont go bigger than class size
    random_img = random.choice(os.listdir(base_dir+ "/train/"+ all_classes[class_num]))
    random_img_path = os.path.join(base_dir+ "/train/" + all_classes[class_num] + "/" + random_img)
    ax.imshow(plt.imread(random_img_path))
    ax.set_title(all_classes[class_num])
    class_num+=1


# In[14]:


#looking good all the random images match the titles


# In[15]:


#Model selection:


# In[16]:


#Xception
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input


# In[17]:


train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    './capstone-project/train',
    target_size=(150, 150),
    batch_size=32
)


# In[18]:


train_ds.class_indices


# In[19]:


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    './capstone-project/validation',
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)


# In[23]:


base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

base_model.trainable = False



inputs = keras.Input(shape=(150, 150, 3))

base = base_model(inputs, training=False)

vectors = keras.layers.GlobalAveragePooling2D()(base)

outputs = keras.layers.Dense(27)(vectors)

model = keras.Model(inputs, outputs)


# In[24]:


learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# In[25]:


history = model.fit(train_ds, epochs=10, validation_data=val_ds)


# In[27]:


plt.plot(history.history['val_accuracy'], label=('val'))
plt.plot(history.history['accuracy'], label=('acc'))
plt.legend()
plt.title('Training acc vs Validation acc')


# In[28]:


#adding inner layers
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

base_model.trainable = False
inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs, training=False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
inner = keras.layers.Dense(500, activation='relu')(vectors)
inner2 = keras.layers.Dense(250, activation='relu')(inner)
outputs = keras.layers.Dense(27,  activation='softmax')(inner2)
model = keras.Model(inputs, outputs)


# In[29]:


optimizer = keras.optimizers.Adam()
loss = keras.losses.CategoricalCrossentropy()
model.compile(optimizer = optimizer, loss = loss, metrics=['accuracy'])


# In[30]:


history = model.fit(train_ds, epochs=10, validation_data=val_ds)


# In[31]:


plt.plot(history.history['val_accuracy'], label=('val'))
plt.plot(history.history['accuracy'], label=('acc'))
plt.legend()
plt.title('Training acc vs Validation acc')


# In[32]:


#VGG16 Model, with additional inner layers


# In[33]:


from tensorflow.keras.applications.vgg16 import VGG16


# In[34]:


train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    './capstone-project/train',
    target_size=(150, 150),
    batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    './capstone-project/validation',
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)


# In[38]:


base_model= VGG16(weights='imagenet', include_top= False, input_shape=(150, 150, 3))
base_model.trainable = False

##################################################
inputs = keras.Input(shape=(150, 150, 3))   
base = base_model(inputs, training= False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
inner = keras.layers.Dense(100, activation= 'relu')(vectors) 
outputs = keras.layers.Dense(27, activation='softmax')(inner) 
model = keras.Model(inputs, outputs)
##################################################

optimizer = keras.optimizers.Adam()
loss = keras.losses.CategoricalCrossentropy()
model.compile(
    optimizer=optimizer, 
    loss = loss, 
    metrics=['accuracy']
)


# In[39]:


history = model.fit(train_ds, epochs=10, validation_data = val_ds)


# In[40]:


plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.title('Train accuracy vs Val accuracy')


# In[41]:


#Not sure about Val being greater than Train accuracy, we will proceed with tuning the XCeption model


# In[45]:


#adjusting the learning rate
def make_model(learning_rate=0.01):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(27)(vectors)
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[46]:


scores = {}

for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(lr)

    model = make_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[lr] = history.history

    print()
    print()


# In[47]:


for lr, hist in scores.items():
    #plt.plot(hist['accuracy'], label=('train=%s' % lr))
    plt.plot(hist['val_accuracy'], label=('val=%s' % lr))

plt.xticks(np.arange(10))
plt.legend()


# In[48]:


#We can see that learning rate of 0.001 is the best one to use


# In[49]:


#adding and tuning more layers
def make_model(learning_rate=0.01, size_inner=100):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    
    outputs = keras.layers.Dense(27)(inner)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[50]:


learning_rate = 0.001

scores = {}

for size in [10, 100, 1000]:
    print(size)

    model = make_model(learning_rate=learning_rate, size_inner=size)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[size] = history.history

    print()
    print()


# In[52]:


for size, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('size=%s' % size))

plt.xticks(np.arange(10))
plt.yticks([0.83, 0.90, 0.95])
plt.legend()


# In[53]:


#we will go with size 1000 since ti achieves optimum level of accuracy the fastest


# In[54]:


#Tuning the dropout
def make_model(learning_rate=0.01, size_inner=1000, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(27)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[56]:


learning_rate = 0.001
size = 1000

scores = {}

for droprate in [0.0, 0.4, 0.8]:
    print(droprate)

    model = make_model(
        learning_rate=learning_rate,
        size_inner=size,
        droprate=droprate
    )

    history = model.fit(train_ds, epochs=12, validation_data=val_ds)
    scores[droprate] = history.history

    print()
    print()


# In[57]:


for droprate, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % droprate))

plt.ylim(0.78, 0.99)
plt.legend()


# In[58]:


#clearly, droprate of 0.0 gives the best result


# In[59]:


#We will now see if simple data augmentation gives us an even better model, accuracy to beat is around 9.60


# In[62]:


train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10,
    zoom_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_ds = train_gen.flow_from_directory(
    './capstone-project/train',
    target_size=(150, 150),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    './capstone-project/validation',
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)


# In[63]:


learning_rate = 0.001
size = 1000
droprate = 0.0

model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=15, validation_data=val_ds)


# In[64]:


hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')

plt.legend()


# In[65]:


#not much improvement so we can drop the augmentation. We will now train a larger model and keep the best one using checkpointing


# In[73]:


def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(27)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


# In[74]:


input_size = 299
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_ds = train_gen.flow_from_directory(
    './capstone-project/train',
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    './capstone-project/validation',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)


# In[75]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_1_{epoch:02d}_{val_accuracy:.3f}.h5.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[76]:


learning_rate = 0.001
size = 1000
droprate = 0.0

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=20, validation_data=val_ds,
                   callbacks=[checkpoint])


# In[ ]:


#We sill use the best model with val accuracy of .97

