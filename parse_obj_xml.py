import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def read_gt(filename, is_det=False):

	ret_dict = {}
	f = open(filename, 'r')
	for line in f:
		l = line.split(',')
		f_no = int(l[0])
		ob_id = int(l[1])
		x1 = float(l[2])
		y1 = float(l[3])
		w = float(l[4])
		h = float(l[5])
		conf = int(l[6])
		x2 = x1 + w
		y2 = y1 + h
		ob_dict = {}
		ob_dict['id'] = ob_id
		ob_dict['bb'] = np.array([x1, y1, x2, y2])
		ob_dict['conf'] = conf
		if f_no in ret_dict:
			ret_dict[f_no].append(ob_dict)
		else:
			ret_dict[f_no] = [ob_dict]

	for key in ret_dict:
		ret_dict[key] = np.array(ret_dict[key])

	save_obj(ret_dict, 'MOT_15_dict')

read_gt('/home/rohit497/16-720/Project/MultiObjectTracking/2DMOT2015/train/ADL-Rundle-6/gt/gt.txt')
s = load_obj('MOT_15_dict')
print(s[1])



