import cv2
from intelligent_placer_lib import Placer as Pl


def check_image(path):
    photo = cv2.imread(path)

    # константы для разбиения фотографий на две половины (используются фотографии размером 1065х800)
    midline = 532
    height = 1065
    width = 800

    polygon = photo[0:midline, 0:width]
    items = photo[midline:height, 0:width]
    poly_contour = Pl.get_contour_polygon(polygon)
    items_contours = Pl.get_contours_items(items)
    size = (height, width)
    result = Pl.placing(size, items_contours, poly_contour)
    return result


print(check_image('C:/Users/79523/PycharmProjects/IntelligentPlacer-develop/data/test19.jpg'))
