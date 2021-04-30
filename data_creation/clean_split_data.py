# Loop through files in split data folder. Only write images with non-empty label files
import os
import shutil

filenames = []
folder = "./split_train"
directory = os.fsencode(folder + "/yolo_labels/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        filenames.append(filename)

i = 0
clean_yolo_dir = folder + "_clean/yolo_labels"
clean_img_dir = folder + "_clean/images"
clean_label_dir = folder + "_clean/labelTxt"
for filename in filenames:
    if os.path.isfile(folder + "/yolo_labels/"+filename) and os.path.getsize(folder + "/yolo_labels/"+filename):
        with open(folder + "/yolo_labels/"+filename) as f:
            num = filename[:-4]
            # label_name = './split_train/labelTxt/' + filename
            yolo_name = folder + '/yolo_labels/' + filename
        #   img_name = './split_train/images/' + num + '.png'
        #   shutil.copy(label_name, clean_label_dir)
        #   shutil.copy(img_name, clean_img_dir)
            shutil.copy(yolo_name, clean_yolo_dir)
        i += 1
        if i%10 == 0:
            print("Finished file {}".format(i))