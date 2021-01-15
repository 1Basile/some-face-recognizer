from PIL import Image
import os
from os import listdir
from os.path import isfile, join

not_scaled_photo = [f for f in listdir(".") if isfile(join(".", f))]

for photo in not_scaled_photo:
    image = Image.open(photo)
    if "_not_scaled" in photo:
        print(photo.split("_")[0])
        os.rename(photo, "{}.png".format(photo.split("_")[0]))
    new_image = image.resize((640, 480))
    new_image.save(photo)
