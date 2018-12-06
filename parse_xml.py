import xml.etree.ElementTree as ET
import numpy as np
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

ret_dict = {}
tree = ET.parse('./cvpr10_tud_stadtmitte/cvpr10_tud_stadtmitte.al')
root = tree.getroot()
for child in root:
	# within annotation tag now
	curr_img = ""
	for annot_child in child:
		if annot_child.tag == "image":
			curr_img = annot_child[0].text
			ret_dict[curr_img] = []

		elif annot_child.tag == "annorect":
			x1 = int(annot_child[0].text)
			y1 = int(annot_child[1].text)
			x2 = int(annot_child[2].text)
			y2 = int(annot_child[3].text)
			ret_dict[curr_img].append(np.array([x1, y1, x2, y2]))

	ret_dict[curr_img] = np.array(ret_dict[curr_img])

save_obj(ret_dict, 'bbox_dict')
d = load_obj('bbox_dict')
print(d['DaMultiview-seq7022.png'])
