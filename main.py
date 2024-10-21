# from neuralfield.script import config
# from neuralfield.network import train
from flet import app

from neuralfield.interface.main import main_windows

# 1034

# data = config.get_configuration()
# cnn_data = data['default']
# print(cnn_data)
# cnn = train.Unet(
#     classes=cnn_data['classes'],
#     output_size=cnn_data['output_size'],
#     sampel_size=cnn_data['sample_size'],
#     epoch= cnn_data['epoch'],
#     count_aug = cnn_data['count_aug']
# )
if __name__ == '__main__':
    app(target=main_windows) 
