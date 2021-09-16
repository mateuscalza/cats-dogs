
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

dataset_home = './'
subdirs = ['treinamento/', 'validacao/']
for subdir in subdirs:
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)

seed(1)
val_ratio = 0.25
src_directory = 'original/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'treinamento/'
	if random() < val_ratio:
		dst_dir = 'validacao/'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		copyfile(src, dst)
