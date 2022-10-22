import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.feature import canny


#  функция, проводящая предварительную проверку на то, что сумма площадей объектов меньше площади многоугольника
def area_check(polygon, items):
    return cv2.contourArea(polygon) > sum([cv2.contourArea(item) for item in items])


#  функция, находящая контуры многоугольника
def get_contour_polygon(img):


#TODO: write a function


#  функция, находящая контуры предметов
def get_contours_items(img):


#TODO: write a function

#  функция, работающая с датасетом (разбивает фотографии на верхнюю и нижнюю часть)
def process_data(path_data='data/*.jpg'):
    raw_data = imread_collection(path_data)
    images, polygons, items = [], [], []

    # проверка на то, что датасет загрузился корректно
    for image in raw_data:
        images.append(rgb2gray(image))
    fig, ax = plt.subplots(3, 4, figsize=(15, 10))
    for i in range(len(images)):
        ax[i // 4][i % 4].imshow(images[i], cmap="gray")
    dataset_plot = os.path.join("work_images", "dataset.png")
    plt.savefig(dataset_plot)

    height = int(images[0].shape[0])
    midline = int(images[0].shape[0] / 2)
    width = int(images[0].shape[1])

    for i in range(len(images)):
        polygon = images[i][0:midline, 0:width]
        polygons.append(polygon)
        item = images[i][midline:height, 0:width]
        items.append(item)

    return polygons, items


# функция, которая по фотографиям датасета после разбиения, возвращает контуры многоугольника и предметов
def contour_processing(polygons, items):


#TODO: write a function

def polygons_plot(polygons):
    fig_up, ax_up = plt.subplots(3, 4, figsize=(15, 10))
    for i in range(len(items_set)):
        ax_up[i // 4][i % 4].imshow(polygons[i], cmap="gray")
    polygon_plot = os.path.join("work_images", "polygon.png")
    plt.savefig(polygon_plot)


def items_plot(items):
    fig_down, ax_down = plt.subplots(3, 4, figsize=(15, 10))
    for i in range(len(items_set)):
        ax_down[i // 4][i % 4].imshow(items[i], cmap="gray")
    item_plot = os.path.join("work_images", "item.png")
    plt.savefig(item_plot)


if __name__ == '__main__':
    polygons_set, items_set = process_data()
    polygons_plot(polygons_set)
    items_plot(items_set)
    contour_processing(polygons_set, items_set)