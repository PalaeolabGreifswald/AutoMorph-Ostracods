#!/usr/bin/env python

import settings
import images
import numpy as np
import subprocess
import sys
import os
import socket
from PIL import Image
from rembg import remove


# Started on 3/21/2014 by Yusu Liu
# code uses base code of PM Hull (20-Oct-13) with updates by B. Dobbins, PMH, and Y. Liu
# converted to python and updated by K. Nelson (2015-Oct)
#updated for Python 3.12 and the use of EDF images by M. Hoehle (April 2025)


def segment(settings_file):

    version = '2016-7-12'

    print(f"Loading settings from {settings_file}...")
    runs = settings.parse(settings_file)

    for i, run in enumerate(runs):

        print(f'Segment - running configuration {i+1} of {len(runs)} from {settings_file}')

        # Get the list of images in the directory
        image_list = images.list_files(run['directory'], run['input_ext'])

        # Set up additional run parameters
        run['image_file_label'] = f'th={run["threshold"]:05.4f}_size={run["minimum_size"]:04.0f}u-{run["maximum_size"]:04.0f}u'
        run['image_label'] = contruct_image_label(run, version)

        if not os.path.exists(run['full_output']):
            os.makedirs(run['full_output'])

        print(f"Found {len(image_list)} images. Starting segmentation...")

        for idx, image_filename in enumerate(image_list, start=1):
            print(f"Processing image {idx}/{len(image_list)}: {os.path.basename(image_filename)}")

            # Detect objects in the current image
            objects, _ = setup_object_boxes(image_filename, run)

            if run['mode'] == 'final':

                if len(objects) > 10000:
                    sys.exit(f'Over 10,000 objects identified in {image_filename}. Stopping to prevent overload.')

                print(f'Saving settings into {run["full_output"]}')
                settings.save(run.copy())

                final(image_filename, objects, run)



def get_git_version():

    gitproc = subprocess.Popen(['git', 'show-ref'], stdout=subprocess.PIPE)
    (stdout, stderr) = gitproc.communicate()

    for row in stdout.decode().split('\n'):
        if 'HEAD' in row:
            hash = row.split()[0]
            break

    return hash


def setup_object_boxes(image_filename, run):

        # Load and resize top-level image
        image = images.load(image_filename, run)

        # Identify all objects based on threshold and size values
        print('INFO: Finding objects')
        objects = images.find_objects(image, run)

        print('INFO: Saving overview image')
        images.save_overview_image(image, objects, image_filename, run)

        return objects, image


def final(orig_filename, box_list, run, image=None):

    if image is None:
        image = images.load(orig_filename, run)

    image_size = np.shape(image)   # [width, height]

    for box_num, box in enumerate(box_list):

        # crop expects [x1, y1, x2, y2], box is [y1, x1, y2, x2]
        crop_box = [box[1], box[0], box[3], box[2]]
        width = crop_box[2] - crop_box[0]
        height = crop_box[3] - crop_box[1]
        image_subsample = images.crop(image, crop_box)
        pil_image = Image.fromarray(image_subsample)
        pil_image = pil_image.convert("RGBA")  # Ensure that alpha channel is present

        no_bg = remove(pil_image)

        # Replaces background witch black
        bg_removed = Image.new("RGBA", no_bg.size, (0, 0, 0, 255))
        bg_removed.paste(no_bg, mask=no_bg.split()[3])  
        
        # Optional: convert to RGB if desired
        final_image = bg_removed.convert("RGB")

        x_percent = float(crop_box[0]) / float(image_size[1]) * 100
        y_percent = float(crop_box[1]) / float(image_size[0]) * 100
        description = f'Object #{box_num+1:05d} of {len(box_list):05d} ({width} x {height} pixels at slide position {x_percent:05.2f} x {y_percent:05.2f})'

        labeled_image_subsample, label = images.label_image(np.array(final_image), orig_filename,
                                                            description, run)

        object_directory = run["full_output"]
        os.makedirs(object_directory, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(orig_filename))[0]
        output_filename = os.path.join(
            run["full_output"],
            f'{base_name}_{box_num+1:03d}.{run["output_ext"]}'
        )

        
        tags = images.add_comment(output_filename, '. '.join(label))
        images.save(labeled_image_subsample, output_filename, tags=tags)


def contruct_image_label(run, version):

    catalog_number = 'None'
    if run['catalog_prefix']:
        if 'IP' in run['unique_id']:  # Special fix for Yale
            catalog_number = f'{run["catalog_prefix"]} {run["unique_id"].split(".")[1]}'
        else:
            catalog_number = f'{run["catalog_prefix"]} {run["unique_id"]}'

    text = []
    text.append(f'{run["units_per_pixel"]:4.2f} {run["unit"]} per pixel | Age and Source:  {run["age"]} from {run["source"]}')
    this_line = f'Processed at {run["location"]}'

    if run['author']:
        this_line += f' by {run["author"]}'
    this_line += f' (Catalog Number: {catalog_number})'

    text.append(this_line)
    text.append(f'CODE VERSION: {version}, PROCESSED ON: {run["timestamp"]}')
    text.append(f'Threshold of {run["threshold"]:4.2f} and size filter of {run["minimum_size"]} - {run["maximum_size"]} {run["unit"]}')
    text.append(f'Directory: {run["subdirectory"]}')

    return text

if __name__ == "__main__":

    if socket.gethostname() == 'tide.geology.yale.edu':
        os.nice(10)

    if len(sys.argv) == 2:
        segment(sys.argv[1])

    else:
        print('Usage: segment <settings_file>')
        sys.exit('Error: incorrect usage')