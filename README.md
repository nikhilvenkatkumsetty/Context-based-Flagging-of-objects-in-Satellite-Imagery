Best Classes for identification: 'ship','large-vehicle','plane','storage-tank'
Small-vehicle appears too much and will bias classification

Full Set (not split):

	train set (304 big images):{'small-vehicle': 79930, 'ship': 18742, 'large-vehicle': 8218, 		'plane': 5528, 'storage-tank': 3733}

	val set (45 big images): {'small-vehicle': 7013, 'ship': 4398, 'plane': 1145,
	'storage-tank': 451, 'large-vehicle': 425}

	test set (45 big images):{'small-vehicle': 24201, 'storage-tank': 1490, 'large-vehicle': 		814, 'plane': 733, 'ship': 659}


Labels are of the form:
'imagesource':imagesource
'gsd':gsd
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
...						
	
Where difficult is whether the particular classification is traditionally hard for classifiers.
If you want to use smaller images with the individual bounding boxes let me know. I can do that pretty easily.

For YOLO labels, the classes are:
0 = 'plane'
1 = 'large-vehicle'
2 = 'ship'
3 = 'storage-tank'
Images with no class have empty text files

Split Train: 4764 (Clean: 3134)
Split Val: 587 (Clean: 385)
Split Test: 795 (Clean: 534)
