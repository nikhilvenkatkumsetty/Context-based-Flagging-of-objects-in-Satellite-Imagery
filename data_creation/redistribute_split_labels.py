# Loop through split folders and redistribute so even in test and val
import os
import shutil
import collections

big_counter = {}
val_names = []
test_names = []
sub_images = {}
directory = os.fsencode("./split_val_clean/yolo_labels/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        val_names.append(filename)
directory = os.fsencode("./split_test_clean/yolo_labels/")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        test_names.append(filename)

i = 0
val_counter = collections.Counter()
for filename in val_names:
    c = collections.Counter()
    with open("./split_val_clean/yolo_labels/" + filename) as f:
        filename_parts = filename.split('_')
        big_img = filename_parts[0]
        small_img = filename_parts[2]+ '__'+ filename_parts[4]+ '___'+ filename_parts[7]
        for line in f:
            words = line.split()
            label = words[0]
            c.update(label)
        val_counter += c
        if big_img not in big_counter:
            big_counter[big_img] = c
        else:
            big_counter[big_img] += c
        if big_img not in sub_images:
            sub_images[big_img] = [small_img]
        else:
            sub_images[big_img].append(small_img)
    i += 1
    if i % 10 == 0:
        print("Finished val file {}".format(i))
i = 0
test_counter = collections.Counter()
for filename in test_names:
    c = collections.Counter()
    with open("./split_test_clean/yolo_labels/" + filename) as f:
        filename_parts = filename.split('_')
        big_img = filename_parts[0]
        small_img = filename_parts[2] + '__' + filename_parts[4] + '___' + filename_parts[7]
        for line in f:
            words = line.split()
            label = words[0]
            c.update(label)
        test_counter += c
        if big_img not in big_counter:
            big_counter[big_img] = c
        else:
            big_counter[big_img] += c
        if big_img not in sub_images:
            sub_images[big_img] = [small_img]
        else:
            sub_images[big_img].append(small_img)
    i += 1
    if i % 10 == 0:
        print("Finished test file {}".format(i))
total_counter = val_counter + test_counter
assignments = {}
new_test_counter = collections.Counter()
new_test_names = []
new_val_counter = collections.Counter()
new_val_names = []
keys = big_counter.keys()
sorted_keys = sorted(keys,key = lambda k: sum(big_counter[k].values()))
i = "val"
prev_diffs = [1000, 1000, 1000, 1000]
for k in reversed(sorted_keys):
    if i == "val":
        best_class_tup = new_test_counter.most_common(1)
        if len(best_class_tup) == 0:
            new_val_names.append(k)
            new_val_counter.update(big_counter[k])
            i = "test"
        else:
            intersection = [k for k in new_test_counter.keys() if k in new_val_counter.keys()]
            if len(intersection) != 0:
                diffs = [new_test_counter[key]-new_val_counter[key] for key in intersection]
                if max(diffs) > 0 & sum([abs(y) for y in diffs]) < sum([abs(x) for x in prev_diffs]):
                    new_val_names.append(k)
                    new_val_counter.update(big_counter[k])
                    prev_diffs = list(diffs)
                else:
                    new_test_names.append(k)
                    new_test_counter.update(big_counter[k])
                    prev_diffs = list([-1*x for x in diffs])
                    i = "test"
            else:
                new_val_names.append(k)
                new_val_counter.update(big_counter[k])
                i = "test"
    elif i == "test":
        best_class_tup = new_val_counter.most_common(1)
        if len(best_class_tup) == 0:
            new_test_names.append(k)
            new_test_counter.update(big_counter[k])
            i = "val"
        else:
            intersection = [k for k in new_test_counter.keys() if k in new_val_counter.keys()]
            if len(intersection) != 0:
                diffs = [new_val_counter[key] - new_test_counter[key] for key in intersection]
                if max(diffs) > 0 & sum([abs(y) for y in diffs]) < sum([abs(x) for x in prev_diffs]):
                    new_test_names.append(k)
                    new_test_counter.update(big_counter[k])
                    prev_diffs = list(diffs)
                else:
                    i = "val"
                    new_val_names.append(k)
                    new_val_counter.update(big_counter[k])
                    prev_diffs = list([-1*x for x in diffs])
            else:
                new_test_names.append(k)
                new_test_counter.update(big_counter[k])
                i = "val"

test_subimages = []
for macro_name in new_test_names:
    for subimage in sub_images[macro_name]:
        test_subimages.append(macro_name+ '__'+subimage)
val_subimages = []
for macro_name in new_val_names:
    for subimage in sub_images[macro_name]:
        val_subimages.append(macro_name + '__' + subimage)
val_size = len(val_subimages)
test_size = len(test_subimages)

# Finally write to a new folder
"""
i = 0
folder = "./split_val"
clean_yolo_dir = folder + "_clean_balanced/yolo_labels"
clean_img_dir = folder + "_clean_balanced/images"
clean_label_dir = folder + "_clean_balanced/labelTxt"
for filename in val_subimages:
    if os.path.isfile("split_clean_combined/yolo_labels/"+filename) and os.path.getsize("split_clean_combined/yolo_labels/"+filename):
        with open("split_clean_combined/yolo_labels/"+filename) as f:
            num = filename[:-4]
            label_name = 'split_clean_combined/labelTxt/' + filename
            yolo_name = 'split_clean_combined/yolo_labels/' + filename
            img_name = 'split_clean_combined/images/' + num + '.png'
            shutil.copy(label_name, clean_label_dir)
            shutil.copy(img_name, clean_img_dir)
            shutil.copy(yolo_name, clean_yolo_dir)
        i += 1
        if i%10 == 0:
            print("Finished val file {}".format(i))
    """
i = 0
folder = "./split_test"
clean_yolo_dir = folder + "_clean_balanced/yolo_labels"
clean_img_dir = folder + "_clean_balanced/images"
clean_label_dir = folder + "_clean_balanced/labelTxt"
for filename in test_subimages:
    if os.path.isfile("split_clean_combined/yolo_labels/"+filename) and os.path.getsize("split_clean_combined/yolo_labels/"+filename):
        with open("split_clean_combined/yolo_labels/"+filename) as f:
            num = filename[:-4]
            label_name = 'split_clean_combined/labelTxt/' + filename
            yolo_name = 'split_clean_combined/yolo_labels/' + filename
            img_name = 'split_clean_combined/images/' + num + '.png'
            shutil.copy(label_name, clean_label_dir)
            shutil.copy(img_name, clean_img_dir)
            shutil.copy(yolo_name, clean_yolo_dir)
        i += 1
        if i%10 == 0:
            print("Finished test file {}".format(i))