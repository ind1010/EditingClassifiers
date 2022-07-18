from pycocotools import coco
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# # import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# # import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# # for evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# unused for now
def obtain_means():
	load_labrador_train = coco.COCO(annotation_file='./data/labrador_train_all.json') # 26,27,17,18,19,20
	# for other images
	# load_labrador_train = coco.COCO(annotation_file='/data/labrador_r02102725.json') # 23
	# load_labrador_train = coco.COCO(annotation_file='/data/labrador_r02103176.json') # 24
	# load_labrador_train = coco.COCO(annotation_file='/data/labrador_r02102965.json') # 94
	# load_labrador_train = coco.COCO(annotation_file='/data/labrador_r02103116.json') # 21
	
	# hard-code test images
	test_images = ['./test/labrador_r02102725.tif','./test/labrador_r02103176.tif', 
	'./test/labrador_r02102965.tif','./test/labrador_r02103116.tif']

	# hard-code image C
	archaeos = load_labrador_train.getAnnIds(imgIds=[17], catIds=[1])
	archaeos_anns = load_labrador_train.loadAnns(ids=archaeos)
	groundtruth = './data/groundtruth_3406.tif' # using pre-loaded binary mask, can also just annotations
	gt_mask = torch.from_numpy(np.array(Image.open(groundtruth))/255).cuda()

	all_precisions = []
	all_recalls = []
	all_ious = []
	for i in tqdm.tqdm(test_images):
		# inference set-up
		im = cv2.imread(i)
		outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

		pred_masks = outputs['instances'].get('pred_masks')
		pred_classes = outputs['instances'].get('pred_classes')

		precisions = []
		recalls = []
		ious = []

		for gt_ann in archaeos_anns:
			gt_instance = torch.from_numpy(load_labrador_train.annToMask(gt_ann)).cuda()
			max_iou = 0
			precision = 0
			recall = 0
			for i, (m,c) in enumerate(zip(pred_masks, pred_classes)):
				if c==0: # hard-coded archaeocyathid class
					pred_mask = m.cuda()
					iou = (torch.sum(torch.logical_and(gt_instance,pred_mask)) / torch.sum(torch.logical_or(gt_instance,pred_mask))).to('cpu')
					if iou > max_iou:
						max_iou = iou
						precision = (torch.sum(torch.logical_and(gt_mask,pred_mask)) / torch.sum(pred_mask)).to('cpu')
						recall = (torch.sum(torch.logical_and(gt_instance,pred_mask)) / torch.sum(gt_instance)).to('cpu')
			ious.append(max_iou)
			precisions.append(precision)
			recalls.append(recall)

		all_precisions.append(np.array(precisions))
		all_recalls.append(np.array(recalls))
		all_ious.append(np.array(ious))

	return all_precisions,all_recalls,all_ious


def display_metrics(precisions=None, recalls=None, ious=None):
	metrics = ['Precision', 'Recall', 'IoU']
	original_metrics = ['original_precisions','original_recalls','original_ious']
	tuned_metrics = ['tuned_nonarchtoredmud3406_precisions','tuned_nonarchtoredmud3406_recalls','tuned_nonarchtoredmud3406_ious']
	for j, (original,tuned) in enumerate(zip(original_metrics, tuned_metrics)):
		with open(original,'rb') as f2:
			e2 = pkl.load(f2)
			f2.close()

		o = np.array([0])
		for c,i in enumerate(e2):
			if c in [0,1,2,3,4,5]:
				continue
			else:
				o = np.append(o,i)

		with open(tuned,'rb') as f2:
			e2 = pkl.load(f2)
			f2.close()

		t = np.array([0])
		for c,i in enumerate(e2):
			if c in [0,1,2,3,4,5]:
				continue
			else:
				t = np.append(t,i)

		print(metrics[j] + '\n------------')
		print('Original mean and standard deviation:')
		print('%f,%f'%(np.mean(np.array(o[o>0])),np.std(np.array(o[o>0]))))
		print('Tuned mean and standard deviation:')
		print('%f,%f'%(np.mean(np.array(t[t>0])),np.std(np.array(t[t>0]))))
		print('\n')


def compute_metrics():
	display_metrics()

