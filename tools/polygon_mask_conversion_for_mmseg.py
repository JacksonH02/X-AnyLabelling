import argparse
import json
import os
import time
import cv2

from PIL import Image
from tqdm import tqdm
from datetime import date

import numpy as np
import matplotlib as plt

import sys

sys.path.append('.')
from anylabeling.app_info import __version__

# ======================================================================= Usage ========================================================================#
#                                                                                                                                                      #
# -------------------------------------------------------------------- mask2poly  ----------------------------------------------------------------------#
# python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --mode mask2poly                                                #
#                                                                                                                                                      #
# -------------------------------------------------------------------- poly2mask  ----------------------------------------------------------------------#
# [option1] python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --mode poly2mask                                      #
# [option2] python tools/polygon_mask_conversion.py --img_path xxx_folder --mask_path xxx_folder --json_path xxx_folder --mode poly2mask               #
#                                                                                                                                                      #
# ======================================================================= Usage ========================================================================#

VERSION = __version__
IMG_FORMATS = ['.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm']


def find_index_by_exact_match(data_list, target_str):
    """Finds the index of the element in the list that exactly matches the target string.

  Args:
      data_list: A list of strings.
      target_str: The string to search for.

  Returns:
      The index of the element that matches the target string, or -1 if not found.

  Raises:
      ValueError: If the target string appears multiple times in the list.
  """
    try:
        return data_list.index(target_str)
    except ValueError:
        return -1  # Return -1 if not found


def parse_files_mmseg(images_path, json_path, masks_path, img_format='.bmp', mask_format='.png'):

    mmseg_path = dict(masks_path=masks_path)
    mmseg_path['imgs_path'] = images_path
    json_files = []
    json_paths = []
    for root, dirs, files in os.walk(json_path):
        for file in files:
            if file.endswith('.json'):
                json_paths.append(os.path.join(root, file))
                json_files.append(file[:-5])
    # print(json_files)
    paired_json_files = []
    paired_img_files = []
    paired_save_path_imgs = []
    paired_save_path_masks = []
    paired_save_path_mask255s = []
    for root, dirs, files in os.walk(images_path):
        for file in files:
            base_name, suffix = os.path.splitext(os.path.basename(file))

            if suffix.lower() not in IMG_FORMATS:
                continue
            # check the match json file and img file
            json_index = find_index_by_exact_match(json_files, base_name)
            if json_index == -1:
                continue
            paired_json_path = json_paths[json_index]

            paired_img_path = os.path.join(root, file)
            relative_path = os.path.relpath(paired_img_path, images_path)
            destination_path_mask = os.path.join(masks_path, 'labels', relative_path[:-4] + mask_format)
            destination_path_mask255 = os.path.join(masks_path, 'labels255', relative_path[:-4] + mask_format)
            destination_path_img = os.path.join(masks_path, 'images', relative_path[:-4] + img_format)

            os.makedirs(os.path.dirname(destination_path_mask), exist_ok=True)
            os.makedirs(os.path.dirname(destination_path_mask255), exist_ok=True)
            os.makedirs(os.path.dirname(destination_path_img), exist_ok=True)
            paired_json_files.append(paired_json_path)
            paired_img_files.append(paired_img_path)
            paired_save_path_imgs.append(destination_path_img)
            paired_save_path_masks.append(destination_path_mask)
            paired_save_path_mask255s.append(destination_path_mask255)
    print(paired_save_path_mask255s)
    return [paired_img_files, paired_json_files, paired_save_path_imgs,
            paired_save_path_masks, paired_save_path_mask255s]


class PolygonMaskConversion():

    def __init__(self, epsilon_factor=0.001):
        self.epsilon_factor = epsilon_factor

    def reset(self):
        self.custom_data = dict(
            version=VERSION,
            flags={},
            shapes=[],
            imagePath="",
            imageData=None,
            imageHeight=-1,
            imageWidth=-1
        )

    def get_image_size(self, image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height

    def mask_to_polygon(self, img_file, mask_file, json_file):
        self.reset()
        binary_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 5:
                continue
            shape = {
                "label": "object",
                "text": "",
                "points": [],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            for point in approx:
                x, y = point[0].tolist()
                shape["points"].append([x, y])
            self.custom_data['shapes'].append(shape)

        image_width, image_height = self.get_image_size(img_file)
        self.custom_data['imagePath'] = os.path.basename(img_file)
        self.custom_data['imageHeight'] = image_height
        self.custom_data['imageWidth'] = image_width

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def polygon_to_mask(self, img_file, mask_file, json_file):

        with open(json_file, 'r') as f:
            data = json.load(f)
        polygons = []
        for shape in data['shapes']:
            points = shape['points']
            polygon = []
            for point in points:
                x, y = point
                polygon.append((x, y))
            polygons.append(polygon)

        image_width, image_height = self.get_image_size(img_file)
        image_shape = (image_height, image_width)
        binary_mask = np.zeros(image_shape, dtype=np.uint8)
        for polygon_points in polygons:
            np_polygon = np.array(polygon_points, np.int32)
            np_polygon = np_polygon.reshape((-1, 1, 2))
            cv2.fillPoly(binary_mask, [np_polygon], color=255)
        cv2.imwrite(mask_file, binary_mask)

    def polygon_to_mask_mmseg(self, img_file, json_file, img_save_path, mask_file, mask255_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        polygons = []
        for shape in data['shapes']:
            points = shape['points']
            polygon = []
            for point in points:
                x, y = point
                polygon.append((x, y))
            polygons.append(polygon)
        image = Image.open(img_file)
        image_width, image_height = image.size
        # image_width, image_height = self.get_image_size(img_file)
        image_shape = (image_height, image_width)
        binary_mask255 = np.zeros(image_shape, dtype=np.uint8)
        binary_mask = np.zeros(image_shape, dtype=np.uint8)
        for polygon_points in polygons:
            np_polygon = np.array(polygon_points, np.int32)
            np_polygon = np_polygon.reshape((-1, 1, 2))
            cv2.fillPoly(binary_mask255, [np_polygon], color=255)
            cv2.fillPoly(binary_mask, [np_polygon], color=1)
        cv2.imwrite(mask255_file, binary_mask255)
        cv2.imwrite(mask_file, binary_mask)
        image.save(img_save_path, format="BMP")


def main():
    parser = argparse.ArgumentParser(description='Polygon Mask Conversion')

    parser.add_argument('--img_path', help='Path to image directory')
    parser.add_argument('--mask_path', help='Path to mask directory')
    parser.add_argument('--json_path', default='', help='Path to json directory')
    parser.add_argument('--epsilon_factor', default=0.001, type=float,
                        help='Control the level of simplification when converting a polygon contour to a simplified version')
    parser.add_argument('--mode', choices=['mask2poly', 'poly2mask'], required=True,
                        help='Choose the conversion mode what you need')
    parser.add_argument('-m', '--mmseg', action='store_true', help='generate files for mmsegmentation training')
    parser.add_argument('--mm_path', default='', help='Path to store mmseg training materials')
    args = parser.parse_args()
    print(f"Starting conversion to {args.mode}...")
    start_time = time.time()

    converter = PolygonMaskConversion(args.epsilon_factor)
    MODE_MMSEG = args.mmseg
    # print(MODE_MMSEG)
    # print('/'.join(last_two_folders))

    if args.mode == "mask2poly":
        file_list = os.listdir(args.mask_path)
        for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='blue'):
            img_file = os.path.join(args.img_path, file_name)
            mask_file = os.path.join(args.mask_path, file_name)
            json_file = os.path.join(args.img_path, os.path.splitext(file_name)[0] + '.json')
            converter.mask_to_polygon(img_file, mask_file, json_file)
    elif args.mode == "poly2mask":
        if MODE_MMSEG:
            imgs, jsons, imgs_save, masks_save, mask255s_save = parse_files_mmseg(args.img_path, args.json_path,
                                                                                  args.mask_path)
            combined_iterables = zip(imgs, jsons, imgs_save, masks_save, mask255s_save)
            for img, json, img_save, mask_save, mask255_save in tqdm(combined_iterables,
                                                                     desc='Converting files', unit='file',
                                                                     colour='blue'):
                converter.polygon_to_mask_mmseg(img, json, img_save, mask_save, mask255_save)
        else:
            os.makedirs(args.mask_path, exist_ok=True)
            file_list = os.listdir(args.img_path)
            for file_name in tqdm(file_list, desc='Converting files', unit='file', colour='blue'):
                base_name, suffix = os.path.splitext(file_name)
                if suffix.lower() not in IMG_FORMATS:
                    continue
                if base_name + ".json" not in file_list:
                    continue
                img_file = os.path.join(args.img_path, file_name)
                if not args.json_path:
                    json_file = os.path.join(args.img_path, base_name + '.json')
                else:
                    json_file = os.path.join(args.json_path, base_name + '.json')
                mask_file = os.path.join(args.mask_path, file_name)
                converter.polygon_to_mask(img_file, mask_file, json_file)

    end_time = time.time()
    print(f"Conversion completed successfully!")
    print(f"Conversion time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
