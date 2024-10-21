
import os
import keras
from keras import Model
import numpy as np

import train

from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter


import numpy as np
import cv2 as cv
import imutils
import cv2



def loading_network(path_model)->Model:
    """
    Функия для загрузки модели
    """
    return keras.models.load_model(path_model)


def rendering_imag(
        cnn_model:Model, 
        classes:int= 4,
        sampel_size:list = (512,512),
        rgb_colors: list= [
            (255, 0, 0), 
            (0, 255, 0), 
            (0, 0, 255), 
            (255, 255, 255)],
        frames:str = "./data/best_img/l4.JPG"
    ) :
    """
    Фукция сегментации изображения при помощи обученной модели Unet
    """

    for filename in frames:
        frame = imread(filename)
        sample = resize(frame, sampel_size)
        
        predict = cnn_model.predict(sample.reshape((1,) +  sampel_size + (3,)))
        predict = predict.reshape(sampel_size + (classes,))
            
        scale = frame.shape[0] / sampel_size[0], frame.shape[1] / sampel_size[1]
        
        frame = (frame / 1.5).astype(np.uint8)
        
        for channel in range(0, classes): 
            contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
            contours = measure.find_contours(np.array(predict[:,:,channel]))
            
            try:
                for contour in contours:
                    rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                            contour[:, 1] * scale[1],
                                            shape=contour_overlay.shape)
                    
                    contour_overlay[rr, cc] = 1        
                
                contour_overlay = dilation(contour_overlay, disk(1))
                frame[contour_overlay == 1] = rgb_colors[channel]
            except Exception as e:
                print(f"{e}----{channel=}----{rgb_colors[channel]=}")

        imsave(f'./test_img/predict/{os.path.basename(filename)}', frame)
        return frame


def img_analysis (
        rgb_colors:list = [],
        
        img_path:str = '.\\data\\img\\*.png',
        visualizal_img:bool = False,
        img = None
    ) -> list:

    """
    Функция для анализа изображения
    """

    name = [
        "дорога",
        "ямы",
        "леса",
        "поля"
    ]

    hsv_min = [
        np.array((45, 40, 163), np.uint8), #з дорога
        np.array((146, 158, 0), np.uint8), #к ямы
        np.array((99, 199, 130), np.uint8), #с леса
        np.array((23, 255, 121), np.uint8) #ж поля
    ]

    hsv_max = [
        np.array((173, 255, 255), np.uint8),
        np.array((255, 255, 255), np.uint8),
        np.array((175, 255, 255), np.uint8),
        np.array((38, 255, 255), np.uint8)
    ]


    arial_list = []

    for i in range(len(hsv_min)):
        

        if img is None:
            img = cv.imread(f"{img_path}")
        
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV )
        thresh = cv.inRange(hsv, hsv_min[i], hsv_max[i] )
        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        

        if visualizal_img:
            __visualization_S_img(img, contours, hierarchy, i)

        print(len(contours))
        print(f" имя категории ={name[i]}")

        if len(contours) < 1:
            arial_list.append(0.0)
        else:
            s = 0.0
            for j in contours:
                count_picksels = cv2.contourArea(j)
                s += __convert_pixel_to_metre(count_picksels)
            if s > 50:
                arial_list.append(s)
            else:
                arial_list.append(0.0)
            print(f"площадь равна = {s}")
    print (arial_list)
    return arial_list



def get_percent(a:float, b:float) -> float:
    """
    Высчитывает процент одного числа в другом числе
    """
    return round(a * 100 / b, 2)


def __visualization_S_img(
            img,
            contours,
            hierarchy,
            i
        ) -> None:
    """
    Функция визуализации фотографии из которой будет получена площадь
    """
    cv.drawContours( img, contours, -1, (255 - i *50 ,255 - i *20, 255 - i *40), 4, cv.LINE_AA, hierarchy, 1 )
    cv.imshow('contours', img)

    cv.waitKey()
    cv.destroyAllWindows()


def __convert_pixel_to_metre(
        count_picksels:float,
        size_img:int = 512,
        metre = 960
        ) -> float:
    """
    Функция расчета занимаймой площади
    """
    if count_picksels == 0:
        return 0.0
    else:
        print(f"{metre**2/size_img**2=}      {count_picksels=}")
        return (metre**2/size_img**2) * count_picksels


if __name__ == '__main__':
    img_analysis(visualizal_img=True)