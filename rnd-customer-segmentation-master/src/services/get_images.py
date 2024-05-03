from collections import OrderedDict
import concurrent.futures
from io import BytesIO
import json
import os
import glob
import multiprocessing
from pathlib import Path
import tempfile
import time

import dask.dataframe as dd
from dask.delayed import delayed
import pandas as pd
from PIL import Image
from PIL import ImageFile
import requests

import constants as constants
from utils import set_up_logging, timeit


logger = set_up_logging(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

CHANNEL_NAME = 'images'
IMAGES_PATH = constants.INPUT_PATH / CHANNEL_NAME


def get_job_status(status_file, data):
    status = {}
    if not status_file.exists():
        for sku, _ in data:
            status[sku] = 'pending'
        with open(status_file, 'w') as fout:
            json.dump(status, fout, sort_keys=True, indent=4)
    with open(status_file, 'r') as fin:
        status = json.loads(fin.read())
    for sku, _ in data:
        if sku not in status:
            status[sku] = 'pending'
    return status


def update_job_status(status, status_file):
    with open(status_file, 'w') as fout:
        json.dump(status, fout, sort_keys=True, indent=4)


def get_img_from_url(img_url, session):
    if not img_url.startswith('http'):
        url_split = img_url.split('/')
        if '' in url_split: url_split.remove('')
        img_url = 'http://' + '/'.join(url_split)
    try:
        response = session.get(img_url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception: # exceptions will be logged
        logger.warning(response + ' - ' + img_url)
        return None


def get_imgs_of_instance(instance, resize=False):
    sku, url, img_folder, session = instance
    urls = [url.replace('1-', f'{i}-') for i in range(1, 5)]
    local_path = img_folder / sku[:2]
    files = []

    for url in urls:
        filename = sku[2:] + '_' + url.split('/')[-1]
        local_img = local_path / filename
        
        if not local_img.exists():
            img = get_img_from_url(url, session)
            if img is not None:
                if resize:
                    img = img.resize((299,299), resample=Image.BILINEAR)
                local_path.mkdir(parents=True, exist_ok=True)
                img.save(local_img, 'jpeg')
                files.append(str(local_img))
            else:
                break
        else:
            files.append(str(local_img))
    return sku, files

        
@timeit
def get_imgs(data, status, status_file):
    try:
        full_list = []
        s = requests.Session()
        for sku, url in data:
            full_list.append((sku, url, IMAGES_PATH, s))
        count = 0
        count_valid_urls = 0
        count_sku_with_images = 0
        n_total = len(full_list)
        t0 = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(get_imgs_of_instance, full_list):
                sku, files = result
                status[sku] = files
                if files:
                    count_sku_with_images += 1
                    count_valid_urls += len(files)
                count+=1
                partial_t = time.time() - t0
                logger.info('Progress: {:0.2f} - {:10.02f} secs  Got: {}/{} ({:0.2f}) skus with valid urls'.format(
                    count/n_total, partial_t, count_sku_with_images, n_total, count_sku_with_images/n_total))
                if count > 0 and count % 500 == 0:
                    update_job_status(status, status_file)
        print()
    except Exception as e:
        print("Error: {}".format(e))
        return 0
    return 1


def get_image_urls():
    df = pd.read_csv(constants.INPUT_PATH / 'urls.gz', compression='gzip')
    return [tuple(x) for x in df.values]


def task(): 
    data = get_image_urls()
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    
    status_file = constants.OUTPUT_PATH / 'status.json'
    status = get_job_status(status_file, data)      

    get_imgs(data, status, status_file)


if __name__ == "__main__":

    logger.info('STARTING get_images.py...')
    task()
    logger.info("...get_images.py HAS ENDED.\n")
