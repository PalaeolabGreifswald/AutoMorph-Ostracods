import configparser
from io import StringIO
from datetime import datetime
import numpy as np
import sys
import os
import getpass


def parse(filename):
    # Set up defaults
    defaults = {
        'pixel_size_x': 1,
        'pixel_size_y': 1,
        'get_coordinates': True,
        'draw_aspect_ratio': True,
        'save_intermediates': False,
        'disk_size_opening': 10,
        'contrast_adjustment': 3,
        'threshold_adjustment': 0,
        'disk_size_smoothing': 20,
        'downsample': True,
        'num_points': 100,
        'run3dmorph': False
    }

    required_list = ['in_directory', 'input_ext', 'out_directory', 'sampleID']
    boolean_list = ['get_coordinates', 'draw_aspect_ratio', 'save_intermediates', 'downsample', 'run3dmorph']
    float_list = ['pixel_size_x', 'pixel_size_y', 'contrast_adjustment', 'threshold_adjustment']
    int_list = ['disk_size_opening', 'disk_size_smoothing', 'num_points']

    settings = defaults
    settings['timestamp'] = datetime.now().strftime('%Y-%m-%d at %H:%M:%S')

    parser = configparser.ConfigParser(allow_no_value=True)
    parser.optionxform = str  # Preserve case

    if os.path.isfile(filename):
        try:
            parser.read(filename)
        except configparser.MissingSectionHeaderError:
            with open(filename, 'r') as f:
                content = '[morph2d]\n' + f.read()
            vfile = StringIO(content)
            parser.read_file(vfile)
    else:
        sys.exit("Error: Cannot open settings file " + filename)

    # Set optional variables
    for setting in parser.options('morph2d'):
        if setting in boolean_list:
            try:
                settings[setting] = parser.getboolean('morph2d', setting)
            except:
                pass
        elif setting in float_list:
            try:
                settings[setting] = parser.getfloat('morph2d', setting)
            except:
                pass
        elif setting in int_list:
            try:
                settings[setting] = parser.getint('morph2d', setting)
            except:
                pass
        else:
            settings[setting] = str(parser.get('morph2d', setting))

    # Check for required parameters
    for required in required_list:
        if required not in settings:
            sys.exit('Error: ' + required + ' must be set in settings file.')

    # Set default output directory if none specified
    if not settings['out_directory']:
        settings['out_directory'] = settings['in_directory'] + os.sep + 'morph2d'

    # Strip trailing separator from input directory
    if settings['in_directory'].endswith(os.sep):
        settings['in_directory'] = settings['in_directory'].rstrip(os.sep)

    return settings


def save(settings):
    write_directory = settings['out_directory']

    parser = configparser.ConfigParser()
    parser.optionxform = str  # Preserve case

    username = getpass.getuser()
    timestamp = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')
    filename = f'settings_morph2d_{username}_{timestamp}.txt'

    if not os.path.exists(write_directory):
        os.makedirs(write_directory)

    section = 'morph2d'
    parser.add_section(section)

    for option, value in settings.items():
        parser.set(section, str(option), str(value))

    with open(os.path.join(write_directory, filename), 'w', encoding='utf-8') as f:
        f.write('# Saved Settings:\n')
        parser.write(f)