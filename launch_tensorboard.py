#!/usr/bin/env python3
# This script takes a flat directory of tensorboard logs and restructures
# it so tensorboard only loads specified experiments.

import shutil
import re
import os
import os.path as pth
import glob
from subprocess import Popen

PORT = 6077
SOURCE_DIR = 'logs/'
TARGET_DIR = 'logs/tboard_events/'
INCLUDE_DEBUG = True


include_patterns = [
                    'exp18.*',
                              ]
if not INCLUDE_DEBUG:
    for i in range(len(include_patterns)):
        include_patterns[i] += '\.\d+$'
include_patterns = [re.compile(pt) for pt in include_patterns]


print(f'refreshing target directory {TARGET_DIR}... ', end='')
shutil.rmtree(TARGET_DIR)
os.makedirs(TARGET_DIR)
print('done')


print('creating symlinks... ', end='')
for name in glob.glob(pth.join(SOURCE_DIR, '*')):
    basename = pth.basename(name)
    for pattern in include_patterns:
        match = pattern.match(basename)
        if match:
            dest = pth.relpath(name, TARGET_DIR)
            src = pth.join(TARGET_DIR, basename)
            os.symlink(dest, src)
print('done')


Popen(f'tensorboard --logdir {TARGET_DIR} --port {PORT}', shell=True).wait()
