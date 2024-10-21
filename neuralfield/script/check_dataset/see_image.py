from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

def image_output(
        image:str = './img/i5.JPG',
        mask:str = './masks/m5.png'
    )->None:
    """
    Отрисовка изображения м маски для него
    """

    image = imread(image)
    mask = imread(mask)
    mask = resize(mask, (mask.shape[0], mask.shape[1]))

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 3), dpi=125)

    ax[0].set_title('Image')
    ax[0].set_axis_off()
    ax[0].imshow(image)
    ax[1].set_title('Mask')
    ax[1].set_axis_off()
    ax[1].imshow(mask * 255 / 7)
    plt. show()
    plt.close()