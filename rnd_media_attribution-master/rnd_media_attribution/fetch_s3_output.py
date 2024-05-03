import os
import re
import time
import datetime
from loguru import logger
# # SHITTY CODE JUST TO BE FAST


def gen_sec_timestamp(string):
    str_to_search = '(?P<year>\d\d\d\d)-(?P<month>\d\d)-(?P<day>\d\d)-(?P<hour>\d\d)-(?P<min>\d\d)-(?P<seconds>\d\d)'
    m = re.search(str_to_search, string)
    datetime.datetime.strptime(m.group(0), "%Y-%m-%d-%H-%M-%S").timetuple()
    dt = datetime.datetime.strptime(m.group(0), "%Y-%m-%d-%H-%M-%S")
    return time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)

logger.info('Starting s3 fetch model output...')
aws_s3_ls_tj = 'aws s3 ls s3://719003640801-media-attribution/training-jobs/'
sys_out = os.popen(aws_s3_ls_tj).read()
sys_out = sys_out.split('\n')
list_of_training_jobs = [ss for s in sys_out for ss in s.split('PRE ')][1::2]
ts_list = {lotj: gen_sec_timestamp(lotj) for lotj in list_of_training_jobs}
ts_list = {k: v for k, v in sorted(ts_list.items(), key=lambda item: item[1])}  # ordering dict
oldest_training_job = list(ts_list.keys())[-1]  # get oldest training_job
aws_s3_path_to_output = aws_s3_ls_tj + oldest_training_job + 'output/'
aws_s3_cp_from = (aws_s3_path_to_output + 'model.tar.gz').replace('ls', 'cp')
aws_s3_cp_to = '~/repositories/rnd_media_attribution/ml/model/attention_model/' + oldest_training_job
aws_s3_cp = aws_s3_cp_from + ' ' + aws_s3_cp_to
logger.info(f'Running {aws_s3_cp}')
for i in range(10,0,-1):
    logger.info(f'Starting aws s3 cp in {i} seconds...')
    time.sleep(1)
logger.info(f'Downloading {oldest_training_job} model output')
os.system(aws_s3_cp)
########################################
# os.chdir('../../../../rnd_media_attribution/notebooks')
chdir = '../ml/model/attention_model/'
# filter = [elt.find('media-attribution') == 0 for elt in os.listdir(chdir)]
# filtered = [i for (i, v) in zip(os.listdir(chdir), filter) if v][0]
# filtered
logger.info(f'Extracting to ml/model/attention_model/ {oldest_training_job}')
os.chdir(chdir + oldest_training_job)
os.system('tar -xzvf model.tar.gz')
os.system('rm model.tar.gz')
logger.info('Done!')