# Loop through all label files and do two things:
# 1) count how many times each class appears
# 2) store the numbers of the images that contain each class
# Then make new train/val/test sets by randomly selecting images for each
# Finally make new text files
import os
import shutil
import glob
import collections
#Loop
c = collections.Counter()
img_classes = {}
filenames = []
directory = os.fsencode("./new_train/labels/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        filenames.append(filename)

i = 0
for filename in filenames:
    with open("./new_train/labels/"+filename) as f:
        imgnum = filename[0:-4]
        class_list = []
        for _ in range(2):
            next(f)
        for line in f:
            words = line.split()
            label = words[-2]
            c.update({label:1})
            if label not in class_list:
                class_list.append(label)
        img_classes[imgnum] = class_list
    i+=1
    if i%10 == 0:
        print("Finished file {}".format(i))


def contains_classes(labels, list):
    label_counter = []
    for label in labels:
        if label in list:
            label_counter.append(1)
        else:
            label_counter.append(0)
    return label_counter

classes = ['ship','large-vehicle','plane','storage-tank']
"""
label_dir = "./new_val/labels"
img_dir = "./new_val/images"
i=0
for the_file in os.listdir(label_dir):
    file_path = os.path.join(label_dir, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

for the_file in os.listdir(img_dir):
    file_path = os.path.join(img_dir, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

for key,val in img_classes.items():
    label_counter = contains_classes(classes,val)
    if sum(label_counter) > 1:
        label_name = './val/labels/'+key+'.txt'
        img_name = './val/images/' +key + '.png'
        shutil.copy(label_name,label_dir)
        shutil.copy(img_name,img_dir)
        print("Copied File {}".format(i))
        i+=1
"""


