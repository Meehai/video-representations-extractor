#!/usr/bin/env python3
import sys
from image_utils import image_draw_polygon, image_draw_rectangle
from image_utils import image_draw_circle, image_resize, image_paste
# from image_utils_pil import image_draw_polygon_pil as image_draw_polygon, image_draw_rectangle_pil as image_draw_rectangle

from PIL import Image
import numpy as np

def main():
    image_cat = np.array(Image.open(sys.argv[1]), dtype=np.uint8)[..., 0:3]
    image_sun = np.array(Image.open(sys.argv[2]), dtype=np.uint8)[..., 0:3]

    image_sun_rsz = image_resize(image_sun, height=image_sun.shape[0] // 3, width=None) # auto-scale
    # paste image_sun over image_cat starting from the most top-left of image_cat where white of image sun is ignored.
    im1 = image_paste(image_cat, image_sun_rsz, top_left=(0, 0), background_color=(255, 255, 255))
    im2 = image_draw_circle(im1, center=(500, 100), radius=10, color=(255, 0, 0), fill=True)
    points = [(100, 800), (150, 800), (150, 850), (200, 850), (100, 900)]
    im3 = image_draw_polygon(im2, points, color=(255, 0, 255), thickness=0.75)
    im4 = image_draw_rectangle(im3, top_left=(500, 850), bottom_right=(650, 950), color=(0, 255, 255), thickness=0.75)
    Image.fromarray(im4).save(sys.argv[3])

if __name__ == "__main__":
    main()
