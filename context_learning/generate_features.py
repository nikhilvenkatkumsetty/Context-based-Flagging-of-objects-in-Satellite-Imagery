import os
import collections
from math import atan2

class featureExtractor:
    def __init__(self, folder):
        # Folder is the folder for the yolo_labels of the
        self.folder = folder
        self.files = []
        self.num_classes = 4
        self.macro_list= {}
        self.extractFiles()

    def extractFiles(self):
        directory = os.fsencode(self.folder + '/yolo_labels/')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                self.files.append(filename[0:-4])
                macro_name = filename[0:5]
                if macro_name not in self.macro_list.keys():
                    self.macro_list[macro_name] = [filename[0:-4]]
                else:
                    self.macro_list[macro_name].append(filename[0:-4])

    def subImageCounts(self,file, useDists = False, useMacro = False, hotFix = True):
        """
        Generates features using surrounding object counts for each object in the provided image
        :param file: The number of the image you want to look at (everything in the name besides the extensions)
                useDists: whether to include the average distance to each class in the feature list
        :return: [(feats, label)]: a list of feature- label pairs, where the features are counts for each class in the image
        excluding one particular object (so there is a pair for each object in the image) plus (optionally)
        the average distance to each class occurrence. The label is the true class of the object
        So feature vector with useDists enabled is [class 0 count, class 1 count,...class 3 count, avg of class 0 dist, ..., avg of class 3 dist]
        """
        # First gather the rows in the yolo_labels file for the given image:
        lines = []
        with open(self.folder + '/yolo_labels/' + file + '.txt') as f:
            for line in f:
                lines.append(line)
        out = []
        for i in range(len(lines)):
            c = collections.Counter()
            words = lines[i].split()
            if hotFix:
                label = int(words[4])
            else:
                label = int(words[0])
            if useDists:
                d = {'0': 0, '1': 0, '2': 0, '3':0}
                if hotFix:
                    x = float(words[0])/1024
                    y = float(words[1])/1024
                else:
                    x = float(words[1])
                    y = float(words[2])
            for j in range(len(lines)):
                if i != j:
                    rowwords = lines[j].split()
                    if hotFix:
                        rowclass = rowwords[4]
                        rowx = float(rowwords[0])/1024
                        rowy = float(rowwords[1])/1024
                    else:
                        rowclass = rowwords[0]
                        rowx = float(rowwords[1])
                        rowy = float(rowwords[2])
                    c.update(rowclass)
                    if useDists:
                        dist = ((rowy - y) ** 2 + (rowx - x) ** 2) ** (0.5)
                        d[rowclass] += dist
            feats = []
            # Add class counts
            for keynum in range(self.num_classes):
                if str(keynum) in c.keys():
                    feats.append(c[str(keynum)])
                else:
                    feats.append(0)
            if useDists:
                for keynum in range(self.num_classes):
                    if str(keynum) in c.keys():
                        feats.append(1/(d[str(keynum)]/c[str(keynum)]))
                    else:
                        feats.append(d[str(keynum)])
            if not useMacro:
                out.append((feats,label))
            else:
                out.append((feats,label,[x,y]))
        return out

    def macroImageCounts(self,file,useDirs = False, useColors = False, useDists = False, hotFix = True):
        """
        Generates features for every object in an image using counts in the overall macro-image. Also has options to use
        mean direction for each class (in radians), mean distance to each class, and average color of the macro-image as features
        :param file: The number of the image you want to look at (everything in the name besides the extensions)
                useDists: whether to include (1/average distance to each class) = eff_{class} in the feature list
                useDirs: whether to include the average direction to each class in the feature list
        :return: [(feats, label)]: a list of feature- label pairs, where the features are counts for each class in the image
        excluding one particular object (so there is a pair for each object in the image) plus (optionally)
        the average distance to each class occurrence. The label is the true class of the object
        So feature vector with all flags enabled is:
         [class 0 count, class 1 count,...class 3 count, eff_0, ..., eff_3, avg dir 0, ... avg dir 3]
        """
        # First gather the rows in the yolo_labels file for the given image:
        count_same_image = self.subImageCounts(file,useDists =True,useMacro = True, hotFix = hotFix)
        comps = file.split('_')
        x0 = float(comps[4])/1024
        y0 = float(comps[7])/1024
        out = []

        for pair in count_same_image:
            feats = pair[0]
            loc = pair[2]
            loc = [loc[0]+x0,loc[1]+y0]
            class_counts = {'0': feats[0],'1': feats[1],'2': feats[2], '3':feats[3]}
            total_dists = {'0': 0, '1': 0, '2': 0, '3': 0}
            total_dirs = {'0': 0, '1': 0, '2': 0, '3': 0}
            if useDists:
                for q in range(4):
                    if feats[q+4] != 0:
                        total_dists[str(q)] = (1/feats[q+4])*feats[q]
            # Loop through all other sub-images with the same macro-image and get features
            sub_images = self.macro_list[file[0:5]]
            for sub in sub_images:
                if sub != file:
                    lines = []
                    with open(self.folder + '/yolo_labels/' + sub + '.txt') as f:
                        for line in f:
                            lines.append(line)
                    sub_comps = sub.split('_')
                    rowx0 = float(sub_comps[4])/1024
                    rowy0 = float(sub_comps[7])/1024
                    for j in range(len(lines)):
                        rowwords = lines[j].split()
                        if hotFix:
                            rowclass = rowwords[4]
                            rowx = float(rowwords[0]) / 1024 + rowx0
                            rowy = float(rowwords[1]) / 1024 + rowy0
                        else:
                            rowclass = rowwords[0]
                            rowx = float(rowwords[1]) + rowx0
                            rowy = float(rowwords[2]) + rowy0
                        class_counts[rowclass] +=1
                        if useDists:
                            dist = ((rowy - loc[1]) ** 2 + (rowx - loc[0]) ** 2) ** (0.5)
                            total_dists[rowclass]+=dist
                        if useDirs:
                            dir = atan2(rowy-loc[1],rowx-loc[0])
                            total_dirs[rowclass]+= dir
            new_feats = []
            # Add class counts
            for keynum in range(self.num_classes):
                if str(keynum) in class_counts.keys():
                    new_feats.append(class_counts[str(keynum)])
                else:
                    new_feats.append(0)
            if useDists:
                for keynum in range(self.num_classes):
                    if total_dists[str(keynum)] != 0 and class_counts[str(keynum)] != 0:
                        new_feats.append(1 / (total_dists[str(keynum)] / class_counts[str(keynum)]))
                    else:
                        new_feats.append(total_dists[str(keynum)])
            if useDirs:
                for keynum in range(self.num_classes):
                    if class_counts[str(keynum)] != 0:
                        new_feats.append(total_dirs[str(keynum)] / class_counts[str(keynum)])
                    else:
                        new_feats.append(total_dirs[str(keynum)])
            out.append([new_feats,pair[1]])
        return out
"""
# Add directionality to macro image objects
# Encode color
# Usage Example
f = featureExtractor('./split_test_clean_balanced')
feat_pairs = f.macroImageCounts('P0128__1__633___0',useDists = True, useDirs = True)
x = 2
"""
