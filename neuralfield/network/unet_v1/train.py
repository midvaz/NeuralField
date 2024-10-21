import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter

from ..script import config

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF" }')

class Unet:
    def __init__(
                self,  
                new_rgb = [255,255,255],
                new_class_name:str="",
                classes:int= 4, 
                output_size:list = (512,512), 
                sampel_size:list = (512,512),
                path_img:str = './data/img',
                path_mask:str = './data/masks',
                epoch:int = 25, 
                
                count_aug = 60
                
                ):
        """
        Метод инициализации необходимых переменных
        """
        self.count_aug = count_aug
        self.classes = classes
        self.output_size = output_size
        self.sampel_size = sampel_size
        self.count_aug = count_aug

        self.path_img = path_img
        self.path_mask = path_mask

        self.epochs = epoch

        self.new_class_name = new_class_name
        self.new_rgb = new_rgb
        

    def __load_images(self, image, mask):
        """
        Метод нормализации данных
        """
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image)
        image = tf.image.resize(image, self.output_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image / 255.0
        
        mask = tf.io.read_file(mask)
        mask = tf.io.decode_png(mask)
        mask = tf.image.rgb_to_grayscale(mask)
        mask = tf.image.resize(mask, self.output_size)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        
        masks = []
        
        for i in range(self.classes):
            masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))
        
        masks = tf.stack(masks, axis=2)
        masks = tf.reshape(masks, self.output_size + (self.classes,))

        return image, masks


    def __augmentate_images(self, image, masks):  
        """
        Метод аугментаций фотографий
        """ 
        random_crop = tf.random.uniform((), 0.3, 1)
        image = tf.image.central_crop(image, random_crop)
        masks = tf.image.central_crop(masks, random_crop)
        
        random_flip = tf.random.uniform((), 0, 1)    
        if random_flip >= 0.5:
            image = tf.image.flip_left_right(image)
            masks = tf.image.flip_left_right(masks)
        
        image = tf.image.resize(image, self.sampel_size)
        masks = tf.image.resize(masks, self.sampel_size)
        
        return image, masks
    

    def __spleet_data(self) -> None:
        """
        Метод деления данных на обучающие и проверочных данных
        """
        self.train_dataset = self.dataset.take(2000).cache()
        self.test_dataset = self.dataset.skip(2000).take(100).cache()
        
        self.train_dataset = self.train_dataset.batch(16)
        self.test_dataset = self.test_dataset.batch(16)


    def make_dataset(self):
        """
        Метод создания датасета
        """
        images = sorted(glob.glob(f'{self.path_img}/*.JPG'))
        masks = sorted(glob.glob(f'{self.path_mask}/*.png'))

        print(f"---------------------------{len(masks)}")
        print(f"---------------------------{len(images)}")

        images_dataset = tf.data.Dataset.from_tensor_slices(images)
        masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

        self.dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

        self.dataset = self.dataset.map(self.__load_images, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = self.dataset.repeat(self.count_aug)
        self.dataset = self.dataset.map(self.__augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)
        
        self.__spleet_data()

        print("Dataset ready")


    def rendering_img_mask(self):
        """
        Метод отривока пяти случайных фотографий
        """
        images_and_masks = list(self.dataset.take(5))

        fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(15, 5), dpi=125)

        for i, (image, masks) in enumerate(images_and_masks):
            ax[0, i].set_title('Image')
            ax[0, i].set_axis_off()
            ax[0, i].imshow(image)
                
            ax[1, i].set_title('Mask')
            ax[1, i].set_axis_off()    
            ax[1, i].imshow(image/1.5)
        
            for channel in range(self.classes):
                contours = measure.find_contours(np.array(masks[:,:,channel]))
                for contour in contours:
                    # иттеация по меткам 
                    ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=1, color=self.colors[channel])
        plt.show()
        plt.close()
    
##########################################################################################

    def __input_layer(self):
        """
        Метод создания входного слой
        """
        return tf.keras.layers.Input(shape=self.sampel_size + (3,))


    def __downsample_block(self, filters, size, batch_norm=True):
        """
        Метод создания кодирующего блока
        """
        initializer = tf.keras.initializers.GlorotNormal()

        result = tf.keras.Sequential()
        
        result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

        if batch_norm:
            result.add(tf.keras.layers.BatchNormalization())
        
        result.add(tf.keras.layers.LeakyReLU())
        return result


    def __upsample_block(self, filters, size, dropout=False):
        """
        Метод создания декодирующего блока
        """
        initializer = tf.keras.initializers.GlorotNormal()
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                            kernel_initializer=initializer, use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())
        
        if dropout:
            result.add(tf.keras.layers.Dropout(0.25))
        result.add(tf.keras.layers.ReLU())
        return result


    def __output_layer(self, size):
        """
        Метод создания выходного слоя
        """
        initializer = tf.keras.initializers.GlorotNormal()
        return tf.keras.layers.Conv2DTranspose(self.classes, size, strides=2, padding='same',
                                            kernel_initializer=initializer, activation='sigmoid')


    def creating_model(self):
        """
        Метод создания модели
        """
        inp_layer = self.__input_layer()

        downsample_stack = [
            self.__downsample_block(64, 4, batch_norm=False),
            self.__downsample_block(128, 4),
            self.__downsample_block(256, 4),
            self.__downsample_block(512, 4),
            self.__downsample_block(512, 4),
            self.__downsample_block(512, 4),
            self.__downsample_block(512, 4),
        ]

        upsample_stack = [
            self.__upsample_block(512, 4, dropout=True),
            self.__upsample_block(512, 4, dropout=True),
            self.__upsample_block(512, 4, dropout=True),
            self.__upsample_block(256, 4),
            self.__upsample_block(128, 4),
            self.__upsample_block(64, 4)
        ]

        out_layer = self.__output_layer(4)

        x = inp_layer

        downsample_skips = []

        for block in downsample_stack:
            x = block(x)
            downsample_skips.append(x)
            
        downsample_skips = reversed(downsample_skips[:-1])

        for up_block, down_block in zip(upsample_stack, downsample_skips):
            x = up_block(x)
            x = tf.keras.layers.Concatenate()([x, down_block])

        out_layer = out_layer(x)

        self.unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)


    def __dice_mc_metric(self, a, b):
        """
        Метод создания метрики оценки нейронной сети
        """
        a = tf.unstack(a, axis=3)
        b = tf.unstack(b, axis=3)
        
        dice_summ = 0
        
        for i, (aa, bb) in enumerate(zip(a, b)):
            numenator = 2 * tf.math.reduce_sum(aa * bb) + 1
            denomerator = tf.math.reduce_sum(aa + bb) + 1
            dice_summ += numenator / denomerator
            
        avg_dice = dice_summ / self.classes
        
        return avg_dice


    def __dice_mc_loss(self, a, b):
        """
        Метод создания функции потерь Dice потери
        """
        return 1 - self.__dice_mc_metric(a, b)


    def __dice_bce_mc_loss(self, a, b):
        """
        Метод комбинированной функции из Dice и binary cross entropy
        """
        return 0.3 * self.__dice_mc_loss(a, b) + tf.keras.losses.binary_crossentropy(a, b)
    

    def build_network(self):
        """
        Метод компиляции модели
        """
        self.unet_like.compile(
                            optimizer='adam', 
                            loss=[self.__dice_bce_mc_loss], 
                            metrics=[self.__dice_mc_metric]
                        )


    def train_network(self):
        """
        Метод обучения модели
        """
        self.unet_like.fit(
            self.train_dataset, 
            validation_data=self.test_dataset, 
            epochs=self.epochs, 
            initial_epoch=0)
    

    def save_model(self):
        """
        Метод сохранения обученной модели
        """
        dir = './model/Unet'
        subfolders = sorted([ f.name for f in os.scandir(dir) if f.is_dir() ])
        next_num = 1
        if len(subfolders) > 0:
            next_num += subfolders[-1][subfolders[-1].rfind('_')+1:]

        self.unet_like.save(f'{dir}/my_modle_{next_num}')


if __name__ == '__main__':

    data = config.get_configuration()

    cnn = Unet(
        new_rgb= data['default']['rgb_colors'],
        new_class_name="",
        classes= data['default']['classes'], 
        output_size=  data['default']['output_size'], 
        sampel_size=  data['default']['sample_size'],
        path_img=  data['path']['img'],
        path_mask=  data['path']['masks'],
        epoch=  data['default']['epoch'], 
                
        count_aug =  data['default']['count_aug']
    )
    cnn.make_dataset()
    cnn.creating_model()
    cnn.build_network()
    cnn.train_network()
    cnn.save_model()


    # data = {
    #         'default': 
    #         {
    #             'classes': 4, 
    #             'colors': ['red', 'lime', 'blue', 'black'], 
    #             'rgb_colors': [[255, 0, 0], 
    #                            [0, 255, 0], 
    #                            [0, 0, 255], 
    #                            [255, 255, 255]], 
    #             'sample_size': [512, 512], 
    #             'output_size': [512, 512]}, 
    #             'path': 
    #             {
    #                 'img': './img/*.JPG', 
    #                 'masks': './masks/*.png', 
    #                 'test_img': './test_img/img/*.jpg', 
    #                 'predict_path': './test_img/predict/'
    #             }
    #         }