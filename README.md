num feature_extraction
EJEMPLO DE USO

>>> from BROFEATURES import KPFeatures

>>> test = KPFeatures("SIFT", "SIFT")

>>> test.diagnose("osde/classes/SegurosMedicosSA.jpg", "osde/examples")


num of images to compare with SegurosMedicosSA: 18

VS smg01:

num of matches for ratio = 0.5: 0

num of matches for ratio = 0.7: 29

num of matches for ratio = 0.9: 367

VS smg02:

num of matches for ratio = 0.5: 3

num of matches for ratio = 0.7: 26

num of matches for ratio = 0.9: 337

. . .

VS smg09:

num of matches for ratio = 0.5: 0

num of matches for ratio = 0.7: 21

num of matches for ratio = 0.9: 333

VS smg10:

num of matches for ratio = 0.5: 1

num of matches for ratio = 0.7: 16

num of matches for ratio = 0.9: 351

. . .

VS smsa01:

num of matches for ratio = 0.5: 1422

num of matches for ratio = 0.7: 1440

num of matches for ratio = 0.9: 1488

VS smsa02:

num of matches for ratio = 0.5: 1481

num of matches for ratio = 0.7: 1490

num of matches for ratio = 0.9: 1526
. . .

VS smsa07:

num of matches for ratio = 0.5: 1525

num of matches for ratio = 0.7: 1531

num of matches for ratio = 0.9: 1551

VS smsa09:

num of matches for ratio = 0.5: 1499

num of matches for ratio = 0.7: 1504

num of matches for ratio = 0.9: 1526

vemos que con SIFT y un ratio de 0.7 es fácil diferenciarlas

entrenamos el modelo con la carpeta "classes", que tiene 1 testigo de cada factura

>>> test.train("osde/classes")

osde/classes/SegurosMedicosSA.jpg num of keypoints: 1584

osde/classes/SMGseguros.jpg num of keypoints: 1366

[INFO] Training... this may take a while.

[INFO] Model created from database containing 2 images.

>>> test.save_model("osde", "osde")  

también podría haberse usado autosave=True en train

luego, cuando se quiera clasificar, cargamos el modelo y usamos classify batch

para el mínimo de matches dejamos el default de 50, aunque podría subirse para

evitar falsos positivos, (eso y el ratio lo ajustamos viendo el diagnóstico)

>>> from BROFEATURES import KPFeatures

>>> test = KPFeatures(model_dir="osde", model_name="osde")

[INFO] Model successfully loaded.

>>> test.classify_batch("osde/examples", ratio=0.7)

[INFO] smsa06.jpg assigned to SegurosMedicosSA (1486 matches)

[INFO] smg05.jpg assigned to SMGseguros (1353 matches)

[INFO] smg09.jpg assigned to SMGseguros (495 matches)

[INFO] smsa05.jpg assigned to SegurosMedicosSA (1503 matches)

[INFO] smsa04.jpg assigned to SegurosMedicosSA (1006 matches)

[INFO] smsa02.jpg assigned to SegurosMedicosSA (1490 matches)

[INFO] smg01.jpg assigned to SMGseguros (167 matches)

[INFO] smg06.jpg assigned to SMGseguros (1354 matches)

[INFO] smg03.jpg assigned to SMGseguros (167 matches)

[INFO] smsa07.jpg assigned to SegurosMedicosSA (1531 matches)

[INFO] smsa03.jpg assigned to SegurosMedicosSA (1509 matches)

[INFO] smsa01.jpg assigned to SegurosMedicosSA (1444 matches)

[INFO] smg07.jpg assigned to SMGseguros (239 matches)

[INFO] smg02.jpg assigned to SMGseguros (240 matches)

[INFO] smsa09.jpg assigned to SegurosMedicosSA (1505 matches)

[INFO] smg04.jpg assigned to SMGseguros (1353 matches)

[INFO] smg10.jpg assigned to SMGseguros (65 matches)

[INFO] smg08.jpg assigned to SMGseguros (1353 matches)


[INFO] Classification Results:

Number of images: 18

SegurosMedicosSA: 8 images

SMGseguros: 10 images

0 images were not classified

vemos que pudo clasificar todas las imágenes con los criterios que le dimos.

en este caso no hubo ningún error
