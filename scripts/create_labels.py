import os
import numpy as np
import  json
import cv2


def create_file(folder_path, type_file='train'):
	text_lines = []
	all_folders = os.listdir(folder_path)
	folders_to_keep = []
	val_folders = []
	save_path = ''
	all_jsons = []
	# if type_file == 'train':
	folders_to_keep = [folder for folder in all_folders if folder not in val_folders]
	train_save_path = '../train.txt'
	# else:
	# folders_to_keep = val_folders
	val_save_path = '../val.txt'

	json_matrix = [[os.path.join(folder_path, l_folder, l_file) for l_file in os.listdir(os.path.join(folder_path, l_folder)) if '.json' in l_file] for l_folder in folders_to_keep]
	json_list = np.array(json_matrix).flatten().tolist()

	for json_f in json_list:
		root_path = json_f.replace(json_f.split("/")[-1], "")
		im_path= ''

		with open (json_f) as jf:
			labels = json.load(jf)

			for file_name, label in labels.items():

				textline = ''
				boxes = []
				regions = label.get("regions")
				im_path = os.path.join(root_path, label.get('filename'))
				relative_im_path = im_path.replace('/home/kartik/Documents/aramco/darknet/', '')

				im = cv2.imread(im_path)

				im_width, im_height = im.shape[1], im.shape[0]

				for i, region in enumerate(regions):
					coordinates = region.get("shape_attributes")
					x, y, width, height = coordinates['x'], coordinates['y'], coordinates['width'], coordinates[
						'height']
					x_center = x + (width/2)
					y_center = y + (height/2)
					x_center, y_center, width, height = x_center/im_width, y_center/im_height, width/im_width, height/im_height
					boxes.extend([str(0) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height)])

				if len(boxes):
					# textline += im_path + " " + str(boxes).replace("[", "").replace("]", "").strip()
					textline += "\n".join(boxes).strip()
					im_save_path = im_path.replace('.jpg', '.txt')
					f = open(im_save_path, 'w')
					f.write(textline)
					f.close()
					text_lines.append(relative_im_path)
					temp = 0


	validation_factor = 0.1

	training_index = int(len(text_lines) * validation_factor)

	validation_lines = text_lines[:training_index]

	training_lines = text_lines[training_index:]

	f = open(train_save_path, 'w')
	for ele in training_lines:
		f.write(ele + '\n')
	f.close()

	f = open(val_save_path, 'w')
	for ele in validation_lines:
		f.write(ele + '\n')
	f.close()
	temp = 0

	temp = 0
# labels_json = [l_file for l_file in os.listdir(l_folder) for l_folder in all_folders]

def create_test_file(folder_path):

	text_lines = []
	test_save_path = '../test.txt'

	images_list = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if '.jpg' in f]


	for im_path in images_list:
		textline = ''
		boxes = []
		relative_im_path = im_path.replace('/home/kartik/Documents/aramco/darknet/', '')

		boxes.extend([str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0)])

		if len(boxes):
			# textline += im_path + " " + str(boxes).replace("[", "").replace("]", "").strip()
			textline += "\n".join(boxes).strip()
			im_save_path = im_path.replace('.jpg', '.txt')
			f = open(im_save_path, 'w')
			f.write(textline)
			f.close()
			text_lines.append(relative_im_path)
			# temp = 0



	f = open(test_save_path, 'w')
	for ele in text_lines:
		f.write(ele + '\n')
	f.close()

def output_images(json_file, save_path):
	with open(json_file) as jf:
		coord_list = json.load(jf)
		for coord_dict in coord_list:
			rectangles = []
			im_path = os.path.join('/home/kartik/Documents/aramco/darknet/', coord_dict['filename'])
			im_save_path = os.path.join(save_path, ''.join(im_path.split('/')[-1]))
			im = cv2.imread(im_path)
			im_width, im_height = im.shape[1], im.shape[0]
			for obj_dict in coord_dict['objects']:
				relative_cords = obj_dict['relative_coordinates']
				x_center, y_center, width, height = relative_cords['center_x'] * im_width, relative_cords['center_y'] * im_height, \
													relative_cords['width'] * im_width, relative_cords['height'] * im_height
				x0,y0, x1, y1 = x_center - (width/2), y_center - (height/2), x_center + (width/2), y_center + (height)/2
				im = cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (255,0,255), -1)
			cv2.imwrite(im_save_path, im)
	pass


if __name__ == '__main__':
	# all_folders = os.listdir("/home/kartik/Downloads/Data-20200416T060849Z-001/Data/completed")
	# create_file(folder_path="/home/kartik/Documents/aramco/darknet/build/darknet/x64/data/obj")
	# create_test_file("/home/kartik/Documents/aramco/darknet/build/darknet/x64/data/test_obj")
	output_images(json_file='/home/kartik/Documents/aramco/darknet/result.json',save_path='/home/kartik/Documents/aramco/darknet/results')
	# create_file(folder_path="/home/kartik/Downloads/Data-20200416T060849Z-001/Data/completed", type_file='val')
	pass
