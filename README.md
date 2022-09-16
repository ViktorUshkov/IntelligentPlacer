# Intelligent Placer

**Описание программы:**

На вход подаётся фотография, на которой изображены несколько предметов и многоугольник, начерченный черным маркером на белом листе бумаги. Intelligent Placer по данным, поданным на вход, должен определить: можно ли разместить все предметы вместе на фотографии в предоставленном многоугольнике?

*Вход:* фотография с объектами и многоугольником (путь до фотографии в файловой системе)

*Выход:* булево значение True (если входные данные корректны и предметы помещаются во многоугольник) или False (если входные данные некорректны, или предметы не помещаются во многоугольник)

**Фотометрические требования:**

•	Формат фотографий - *.jpg

•	Направление камеры – перпендикулярно плоскости, на которой расположены предметы и лист с многоугольником

•	Предметы и их границы не должны сливаться с поверхностью

**Требования к многоугольнику и объектам:** 

•	Многоугольник должен быть замкнутым и выпуклым

•	Многоугольник чертится черным маркером на белом листе бумаги

•	Не разрешается использование объектов, не указанных в папке items

•	Один объект не может присутствовать на фотографии более одного раза

•	Лист бумаги с многоугольником и объекты целиком помещаются на фотографии

•	Объекты на фотографии не пересекаются друг с другом и не пересекаются с многоугольником


**Фотографии предметов и поверхности**

[предметы и поверхность](https://github.com/ViktorUshkov/IntelligentPlacer/tree/develop/items)

**Примеры входных фотографий**

[размеченные примеры](https://github.com/ViktorUshkov/IntelligentPlacer/blob/develop/MarkedUpDataTests.md)
