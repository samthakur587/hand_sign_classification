import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 ,MobileNetV3Large,MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D,Dropout,Dense,Add
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip,RandomRotation
from tensorflow.keras.preprocessing import image_dataset_from_directory
class sign_model():
    def load_data(self,files):
        hf = h5py.File(files,'r')
        key = list(hf.keys())
        classes = list(hf.get(key[0]))
        data_x = np.array(hf.get(key[1]))
        data_y = np.array(hf.get(key[2]))
        examples = data_x
        labels = data_y
        dataset = tf.data.Dataset.from_tensor_slices((examples,labels))
        BATCH_SIZE = 32
        SHUFFLE_BUFFER_SIZE = 100
        train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE,seed=0).batch(BATCH_SIZE)
        return classes,train_dataset
    def onehot(self,num_class,y):
        x = np.eye(num_class)
        y_pred = x[:,y]
        return y_pred
    def data_augmenter(self):
        data_augment = tf.keras.Sequential()
        data_augment.add(RandomFlip('horizontal'))
        data_augment.add(RandomRotation(0.2))
        return data_augment
    def show_image(self,dataset):
        plt.figure(figsize=(10,10))
        for image,labels in dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3,3,i+1)
                plt.imshow(image[i].numpy())
                plt.axis('off')
        plt.show()
    def show_augmented_image(self,dataset):
        for image , labels in dataset.take(1):
            aug = self.data_augmenter()
            plt.figure(figsize=(10,10))
            img = image[0]
            for i in range(9):
                ax = plt.subplot(3,3,i+1)
                augmented_image = aug(tf.expand_dims(img,0))
                plt.imshow(augmented_image[0]/255)
                plt.axis('off')
            plt.show()
    def mobile_net_v2(self,image_shape, data_augmentation,preproces_input,fine_tune=78):
        input_shape = image_shape
        base_model = MobileNetV2(input_shape=input_shape,include_top=False,weights='imagenet')
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune]:
                layer.trainable = False
        inputs = tf.keras.Input(shape=input_shape)
        x  = data_augmentation(inputs)
        x = preproces_input(x)
        x = base_model(x,training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(6,activation='softmax')(x)
        model = tf.keras.Model(inputs,outputs)
        return model
    def mobile_net_v3_large(self,image_shape, data_augmentation,fine_tune=200):
        input_shape = image_shape
        base_model = MobileNetV3Large(input_shape=input_shape,include_top=False,weights='imagenet')
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune]:
                layer.trainable = False
        inputs = tf.keras.Input(shape=input_shape)
        x  = data_augmentation(inputs)
        x = base_model(x,training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(6,activation='softmax')(x)
        model = tf.keras.Model(inputs,outputs)
        return model
    def mobile_net_v3_small(self,image_shape, data_augmentation,fine_tune=160):
        input_shape = image_shape
        base_model = MobileNetV3Small(input_shape=input_shape,include_top=False,weights='imagenet')
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune]:
                layer.trainable = False
        inputs = tf.keras.Input(shape=input_shape)
        x  = data_augmentation(inputs)
        x = base_model(x,training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(6,activation='softmax')(x)
        model = tf.keras.Model(inputs,outputs)
        return model
    def plot_accuracy(self,hist,hist1,hist2):
        plt.figure(figsize=(10,10))
        f , (ax1,ax2) = plt.subplots(2,1)
        ax1.plot(hist.history['accuracy'])
        ax1.plot(hist1.history['accuracy'])
        ax1.plot(hist2.history['accuracy'])
        ax1.set_title('training accuracy of mobilenet')
        ax1.set_ylabel('training_accuracy')
        ax1.set_xlabel('epoch')
        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5)
        ax2.plot(hist.history['val_accuracy'])
        ax2.plot(hist1.history['val_accuracy'])
        ax2.plot(hist2.history['val_accuracy'])
        ax2.set_title('validation accuracy mobilenet')
        ax2.set_ylabel('validation_accuracy')
        ax2.set_xlabel('epoch')
        ax2.grid(color = 'green', linestyle = '--', linewidth = 0.5)
        f.legend(['mobile_net_v2(78)', 'mobile_net_v3_large(200)','mobile_net_v3_small(160)'],loc=7)
        f.tight_layout()
        f.subplots_adjust(right=0.75)
        plt.savefig('accuracy.png', bbox_inches='tight')
        plt.show()
    def plot_loss(self,hist,hist1,hist2):
        plt.figure(figsize=(10,10))
        f , (ax1,ax2) = plt.subplots(2,1)
        ax1.plot(hist.history['loss'])
        ax1.plot(hist1.history['loss'])
        ax1.plot(hist2.history['loss'])
        ax1.set_title('training loss mobilenet')
        ax1.set_ylabel('training_loss')
        ax1.set_xlabel('epoch')
        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5)
        ax2.plot(hist.history['val_loss'])
        ax2.plot(hist1.history['val_loss'])
        ax2.plot(hist2.history['val_loss'])
        ax2.set_title('validation loss mobilenet')
        ax2.set_ylabel('validation_loss')
        ax2.set_xlabel('epoch')
        ax2.grid(color = 'green', linestyle = '--', linewidth = 0.5)
        f.legend(['mobile_net_v2(78)', 'mobile_net_v3_large(200)','mobile_net_v3_small(160)'],loc=7)
        f.tight_layout()
        f.subplots_adjust(right=0.75)
        plt.savefig('loss.png', bbox_inches='tight')
        plt.show()
    def plot_performance(self,run_time,hist,hist1,hist2):
        tA = [int(np.max(hist.history['accuracy'])*100) ,int(np.max(hist1.history['accuracy'])*100) ,int(np.max(hist.history['accuracy'])*100)]
        vA = [int(np.max(hist.history['val_accuracy'])*100) ,int(np.max(hist1.history['val_accuracy'])*100) ,int(np.max(hist.history['val_accuracy'])*100)]
        R = run_time
        barWidth = 0.25
        br1 = np.arange(len(R))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        plt.bar(br1, R, color ='darksalmon', width = barWidth,
            edgecolor ='grey', label ='run_time')
        plt.bar(br2, tA, color ='plum', width = barWidth,
            edgecolor ='grey', label ='training_accuracy')
        plt.bar(br3,vA, color ='darkturquoise', width = barWidth,
            edgecolor ='grey', label ='validation_accuracy')
        plt.xlabel('models', fontweight ='bold', fontsize = 15)
        plt.ylabel('accuracy & Run time', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(R))],
            ['mobile_net_v2(78)', 'mobile_net_v3_large(200)','mobile_net_v3_small(160)'],rotation=30)
        for i in range(len(R)):
            plt.text(i, R[i]//2, R[i], ha = 'center',rotation=45)
            plt.text(i+barWidth, tA[i]//2, tA[i], ha = 'center',rotation=45)
            plt.text(i+2*barWidth, vA[i]//2, vA[i], ha = 'center',rotation=45)
        plt.legend(title='models',title_fontsize=30,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75)
        plt.tight_layout()
        plt.show()
    def prediction(model1,model2,model3,test_dataset,image_num=5):
        for image , label in test_dataset.take(1):
            y_pred_1 = model1.predict(np.expand_dims(image[image_num], axis=0))
            y_pred_2 = model2.predict(np.expand_dims(image[image_num], axis=0))
            y_pred_3 = model3.predict(np.expand_dims(image[image_num], axis=0))
            y1 = ['%.2f' % elem for elem in y_pred_1[0]]
            y2 = ['%.2f' % elem for elem in y_pred_2[0]]
            y3 = ['%.2f' % elem for elem in y_pred_3[0]]
            barWidth = 0.25
            br1 = np.arange(len(y_pred_1))
            br2 = [x + barWidth for x in br1]
            br3 = [x + barWidth for x in br2]
            f , (ax1,ax2) = plt.subplots(1,2)
            ax1.imshow(image[image_num].numpy())
            ax1.axis('off')
            ax1.set_title('test image')
            ax2.bar(br1, y1, color ='darksalmon', width = barWidth,
                edgecolor ='grey', label ='mobilenet_v2')
            ax2.bar(br2, y2, color ='plum', width = barWidth,
                edgecolor ='grey', label ='MobileNetV3Large')
            ax2.bar(br3,y3, color ='darkturquoise', width = barWidth,
                edgecolor ='grey', label ='MobileNetV3Small')
            ax2.set_xlabel('classes', fontweight ='bold', fontsize = 15)
            ax2.set_ylabel('probability of classes', fontweight ='bold', fontsize = 15)
            ax2.set_xticks([r + barWidth for r in range(1)],
                [np.argmax(y1)],rotation=30)
            ax2.legend(title='models',title_fontsize=30,loc='center', bbox_to_anchor=(0.65, 1.25))
            f.subplots_adjust(right=0.75)
            f.tight_layout()
            plt.show()
def main():
    run_time = []
    model = sign_model()
    classes,train_dataset = model.load_data('train_signs.h5')
    classes,test_dataset = model.load_data('test_signs.h5')
    model.show_image(test_dataset)
    data_augmentation = model.data_augmenter()
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    preproces_input = tf.keras.applications.mobilenet_v2.preprocess_input
    model.show_augmented_image(train_dataset)
    t1 = time.time()
    model1 = model.mobile_net_v2((64,64,3),data_augmentation,preproces_input)
    base_learning_rate = 0.0001
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    history = model1.fit(train_dataset,epochs=15,validation_data=test_dataset)
    model1.evaluate(test_dataset)
    t2 = time.time()
    run_time.append(int(t2-t1))

    model2 = model.mobile_net_v3_large((64,64,3),data_augmentation)
    base_learning_rate = 0.0001
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    history1 = model2.fit(train_dataset,epochs=15,validation_data=test_dataset)
    model2.evaluate(test_dataset)
    t3 = time.time()
    run_time.append(int(t3-t2))
    model3 = model.mobile_net_v3_small((64,64,3),data_augmentation)
    base_learning_rate = 0.0001
    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
    history2 = model3.fit(train_dataset,epochs=15,validation_data=test_dataset)
    model3.evaluate(test_dataset)
    t4 = time.time()
    run_time.append(int(t4-t3))
    model.plot_accuracy(history,history1,history2)
    model.plot_loss(history,history1,history2)
    model.plot_performance(run_time,history,history1,history2)
    model.prediction(model1,model2,model3,test_dataset)

    model1.save('model_1.h5')
    model2.save('model_2.h5')
    model3.save('model_3.h5')
if __name__== "__main__":
    main()
