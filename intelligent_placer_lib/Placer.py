import numpy as np
import cv2
import os
from skimage.feature import canny
import rotate_contour as rot
import matplotlib.pyplot as plt


#  функция, загружающая фотографии
def get_images(path):
    data = []
    for file in os.listdir(path):
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            img = cv2.imread(os.path.join(path, file))
            data.append((img, file))
    return data


#  функция, занимающаяся обработкой фото с многоугольником перед поиском контура
def polygon_preparation(polygon_img):
    result = cv2.cvtColor(polygon_img, cv2.COLOR_BGR2GRAY)
    result = canny(result, sigma=0.3).astype(np.uint8)
    return result


#  функция, занимающаяся обработкой фото с предметами перед поиском контура
def items_preparation(items_img):
    result = cv2.cvtColor(items_img, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(result, (3, 5), cv2.BORDER_DEFAULT)
    result = canny(result, sigma=0.425).astype(np.uint8)
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
    img = items_preparation(image)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# функция, выполняющая сдвиг контура на величины dx, dy
def contour_shift(contour, dx, dy):
    return contour + [int(dx), int(dy)]


# функция, проверяющая, поместился ли предмет в многоугольник с помощью битовой маски
def is_fit(contour, masks_image):
    contour_image = masks_image.copy()
    # делаем маску объекта и накладываем на маску полигона
    cv2.fillPoly(contour_image, [contour], (255, 255, 255))
    # с помощью логического XOR обнаруживаем области предмета, которые выходят за маску
    # полигона или накладываются на уже расположенные объекты в нём. Если площадь таких областей равна
    # нулю, то предмет будет расположен
    return np.all(np.logical_xor(np.logical_not(contour_image), masks_image))


# функция, пытающаяся найти размещение одного предмета во многоугольнике с помощью
# обхода многоугольника с заявленным шагом step
def place_item(coords, masks_image, contour, x_axis_step, y_axis_step, angle_step):
    # проход по оси y
    for y_place in range(coords[0], coords[1], y_axis_step):
        # проход по оси x
        for x_place in range(coords[2], coords[3], x_axis_step):
            if is_fit(contour, masks_image):
                return True, contour

            # если предмет не вместился в нынешней точке сразу, пробуем его поворачивать
            angle = angle_step
            while angle < 360:
                rot_contour = rot.rotate_contour(contour, angle)
                angle += angle_step
                if is_fit(rot_contour, masks_image):
                    return True, rot_contour

            contour = contour_shift(contour, x_axis_step, 0)
        contour = contour_shift(contour, coords[2] - coords[3], y_axis_step)
    return False, contour


# функция, сравнивающая радиусы описанной окружности предмета и описанной окружности многоугольника
def radius_check(polygon, items):
    items_lesser = True
    (_, _), poly_r = cv2.minEnclosingCircle(polygon)
    for item in items:
        (_, _), item_r = cv2.minEnclosingCircle(item)
        if item_r > poly_r:
            return False

    return items_lesser


#  функция, проводящая предварительную проверку на то, что сумма площадей объектов меньше площади многоугольника
def area_check(polygon, items):
    return cv2.contourArea(polygon) > sum([cv2.contourArea(item) for item in items])


# основная функция плейсера
def placing(image_size, items, polygon):
    # проверка на наличие контуров многоугольника и контуров предметов
    if len(polygon) == 0 or len(items) == 0:
        return False
    # сортировка контуров по признаку убывания радиуса описанной окружности
    items = sorted(items, key=lambda x: cv2.minEnclosingCircle(x)[1], reverse=True)
    # проверка на невозможность размещения по радиусу описанной окружности и сумме площадей
    if not radius_check(polygon, items) or not area_check(polygon, items):
        return False

    # создаем изображение, на котором будем размещать маски
    masks_image = np.zeros((image_size[0], image_size[1], 3), np.uint8)
    # рисуем маску полигона
    cv2.fillPoly(masks_image, [polygon], (255, 255, 255))

    # определяем контур прямоугольника через ограничивающий его прямоугольник
    poly_x, poly_y, poly_w, poly_h = cv2.boundingRect(polygon)
    # выбор шагов перебора (angle_step в градусах)
    x_axis_step, y_axis_step, angle_step = int(poly_w / 10), int(poly_h / 10), 45

    for item in items:
        # перенос контура объекта к контуру многоугольника
        item_x, item_y, item_w, item_h = cv2.boundingRect(item)
        item = contour_shift(item, poly_x - item_x, poly_y - item_y)
        # пытаемся найти место для айтема
        coords = (poly_y + int(item_h / 2),           # y0
                  poly_y + poly_h - int(item_h / 2),  # y1
                  poly_x + int(item_w / 2),           # x0
                  poly_x + poly_w - int(item_w / 2))  # x1

        result, item = place_item(coords, masks_image, item, x_axis_step, y_axis_step, angle_step)
        # если не обнаружили места - выходим, возвращая False
        if not result:
            return False
        # если нашли, наносим маску объекта на найденное место во многоугольнике и обрабатываем следующие предметы
        cv2.fillPoly(masks_image, [item], (0, 0, 0))
        plt.imshow(masks_image)
        plt.show()
    # если распределили все предметы, возвращаем True
    return True
