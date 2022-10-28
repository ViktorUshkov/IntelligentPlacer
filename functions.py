import numpy as np
import cv2
import os
from skimage.feature import canny


#  функция, загружающая фотографии
def get_images(path):
    data = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            img = cv2.imread(os.path.join(path, file))
            data.append((img, file))
    return data


#  функция, занимающаяся обработкой фото с многоугольником перед поиском контура
def polygon_preparation(polygon):
    result = cv2.cvtColor(polygon, cv2.COLOR_BGR2GRAY)
    result = canny(result, sigma=0.3).astype(np.uint8)
    return result


#  функция, занимающаяся обработкой фото с предметами перед поиском контура
#  TODO: подобрать более оптимальные параметры
def item_preparation(item):
    result = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(result, (5, 5), cv2.BORDER_DEFAULT)
    result = canny(result, sigma=0.3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


#  функция, возвращающая контур полигона
def get_contour_polygon(image):
    img = polygon_preparation(image)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return tuple()
    return max(contours, key=lambda x: cv2.contourArea(x))


#  функция, возвращающая контуры предметов
def get_contours_items(image):
    img = item_preparation(image)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


#  функция, проводящая предварительную проверку на то, что сумма площадей объектов меньше площади многоугольника
def area_check(polygon, items):
    return cv2.contourArea(polygon) > sum([cv2.contourArea(item) for item in items])


#  функция, определяющая, войдут ли предметы во многоугольник
def can_fit(polygon, items):
    return len(polygon) and len(items) and area_check(polygon, items)  # пока просто проверка на площади и наличие контуров
