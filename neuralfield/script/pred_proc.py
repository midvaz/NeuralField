"""
Модуль для препроцессинга и постпроцессинга снимков
(для передачи в нейросеть и последующей работы с сегментацией).
"""

import numpy as np
from skimage.morphology import opening, remove_small_objects
from skimage.util import view_as_windows
from PIL import Image
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape

class Preprocessing:
    def __init__(self, r, g, b, nir):
        self.r = r
        self.g = g
        self.b = b
        self.nir = nir

    def calc_indices(self):
        # NDVI = (NIR – R)/(NIR + R), GNDVI = (NIR – G)/(NIR + G)
        ndvi = (self.nir - self.r) / np.clip(self.nir + self.r, 1e-6, None)
        gndvi = (self.nir - self.g) / np.clip(self.nir + self.g, 1e-6, None)
        self.ndvi = ndvi
        self.gndvi = gndvi
        return ndvi, gndvi

    def stack_channels(self):
        ndvi, _ = self.calc_indices()
        # Тензор: R, G, B, NIR, NDVI
        stacked = np.stack([self.r, self.g, self.b, self.nir, ndvi], axis=-1)
        return stacked

    def robust_minmax(self, img, pmin=2, pmax=98):
        # Robust Min-Max нормализация (формула со слайда)
        mi = np.percentile(img, pmin)
        ma = np.percentile(img, pmax)
        out = np.clip(img, mi, ma)
        norm = (out - mi) / (ma - mi + 1e-8)
        return norm

    def normalize(self, tensor):
        # Нормализация всех каналов
        return np.stack([self.robust_minmax(tensor[...,i]) for i in range(tensor.shape[-1])], axis=-1)

    def tile(self, tensor, tile_size=512):
        # Тайлинг для ViT
        return view_as_windows(tensor, (tile_size, tile_size, tensor.shape[-1]), step=tile_size)[...,0]

class Postprocessing:
    def __init__(self, mask, transform=None, crs=None):
        self.mask = mask
        self.transform = transform
        self.crs = crs

    def denoise(self, min_size=100):
        # МОРФОЛОГИЯ и удаление шумов
        cleared = opening(self.mask)
        cleaned = remove_small_objects(cleared.astype(bool), min_size=min_size)
        return cleaned.astype(np.uint8)

    def vectorize(self, class_id=1):
        # Векторизация через rasterio.features.shapes
        mask_bin = (self.mask == class_id).astype(np.uint8)
        shapes = rasterio.features.shapes(mask_bin, transform=self.transform)
        polygons = [shape(s) for s, v in shapes if v == 1]
        gdf = gpd.GeoDataFrame({'geometry': polygons})
        if self.crs: gdf.set_crs(self.crs, inplace=True)
        gdf['area'] = gdf.geometry.area
        gdf['perimeter'] = gdf.geometry.length
        return gdf

    def smooth(self, gdf, tolerance=0.2):
        # Упрощение и сглаживание контуров
        gdf['geometry'] = gdf.buffer(0).simplify(tolerance)
        return gdf

    def export_geojson(self, gdf, path):
        gdf.to_file(path, driver='GeoJSON')

# === Пример использования ===
if __name__ == '__main__':
    # Загрузка каналов (пример, подставьте путь к своим данным)
    img = np.array(Image.open("your_multispectral_image.tif"))  # shape: H x W x 4
    r, g, b, nir = img[...,0], img[...,1], img[...,2], img[...,3]

    pre = Preprocessing(r, g, b, nir)
    tensor = pre.stack_channels()
    norm_tensor = pre.normalize(tensor)
    tiles = pre.tile(norm_tensor, tile_size=512)

    # Допустим, у нас есть маска после нейросети:
    mask = np.array(Image.open("your_segmentation_mask.png"))

    post = Postprocessing(mask)
    mask_clean = post.denoise()
    gdf = post.vectorize()
    gdf = post.smooth(gdf)
    post.export_geojson(gdf, "output.geojson")
