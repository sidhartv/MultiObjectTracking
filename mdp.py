import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from parse_obj_xml import load_obj
import copy
import os
import statistics
import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt

k = 4

images = ['000001.jpg', '000002.jpg', '000003.jpg', '000004.jpg', '000005.jpg', 
'000006.jpg', '000007.jpg', '000008.jpg', '000009.jpg', '000010.jpg', 
'000011.jpg', '000012.jpg', '000013.jpg', '000014.jpg', '000015.jpg', 
'000016.jpg', '000017.jpg', '000018.jpg', '000019.jpg', '000020.jpg', 
'000021.jpg', '000022.jpg', '000023.jpg', '000024.jpg', '000025.jpg', 
'000026.jpg', '000027.jpg', '000028.jpg', '000029.jpg', '000030.jpg',
'000031.jpg', '000032.jpg', '000033.jpg', '000034.jpg', '000035.jpg', 
'000036.jpg', '000037.jpg', '000038.jpg', '000039.jpg', '000040.jpg', 
'000041.jpg', '000042.jpg', '000043.jpg', '000044.jpg', '000045.jpg',
'000046.jpg', '000047.jpg', '000048.jpg', '000049.jpg', '000050.jpg',
'000051.jpg', '000052.jpg', '000053.jpg', '000054.jpg', '000055.jpg', 
'000056.jpg', '000057.jpg', '000058.jpg', '000059.jpg', '000060.jpg',
'000061.jpg', '000062.jpg', '000063.jpg', '000064.jpg', '000065.jpg', 
'000066.jpg', '000067.jpg', '000068.jpg', '000069.jpg', '000070.jpg',
'000071.jpg', '000072.jpg', '000073.jpg', '000074.jpg', '000075.jpg',
'000076.jpg', '000077.jpg', '000078.jpg', '000079.jpg', '000080.jpg',
'000081.jpg', '000082.jpg', '000083.jpg', '000084.jpg', '000085.jpg', 
'000086.jpg', '000087.jpg', '000088.jpg', '000089.jpg', '000090.jpg', 
'000091.jpg', '000092.jpg', '000093.jpg', '000094.jpg', '000095.jpg', 
'000096.jpg', '000097.jpg', '000098.jpg', '000099.jpg', '000100.jpg', 
'000101.jpg', '000102.jpg', '000103.jpg', '000104.jpg', '000105.jpg', 
'000106.jpg', '000107.jpg', '000108.jpg', '000109.jpg', '000110.jpg', 
'000111.jpg', '000112.jpg', '000113.jpg', '000114.jpg', '000115.jpg', 
'000116.jpg', '000117.jpg', '000118.jpg', '000119.jpg', '000120.jpg', 
'000121.jpg', '000122.jpg', '000123.jpg', '000124.jpg', '000125.jpg', 
'000126.jpg', '000127.jpg', '000128.jpg', '000129.jpg', '000130.jpg', 
'000131.jpg', '000132.jpg', '000133.jpg', '000134.jpg', '000135.jpg', 
'000136.jpg', '000137.jpg', '000138.jpg', '000139.jpg', '000140.jpg', 
'000141.jpg', '000142.jpg', '000143.jpg', '000144.jpg', '000145.jpg', 
'000146.jpg', '000147.jpg', '000148.jpg', '000149.jpg', '000150.jpg', 
'000151.jpg', '000152.jpg', '000153.jpg', '000154.jpg', '000155.jpg', 
'000156.jpg', '000157.jpg', '000158.jpg', '000159.jpg', '000160.jpg', 
'000161.jpg', '000162.jpg', '000163.jpg', '000164.jpg', '000165.jpg', 
'000166.jpg', '000167.jpg', '000168.jpg', '000169.jpg', '000170.jpg', 
'000171.jpg', '000172.jpg', '000173.jpg', '000174.jpg', '000175.jpg', 
'000176.jpg', '000177.jpg', '000178.jpg', '000179.jpg']

def get_image(index):
    img_file = '/home/rohit497/16-720/Project/MultiObjectTracking/2DMOT2015/train/TUD-Stadtmitte/img1/' + images[index]
    img = cv2.imread(img_file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class MDP(object):
    def __init__(self, detection, trained_svm, gts):
        self.state_type = 'active'
        self.detection = detection
        self.overlaps = []
        self.gts = gts # list of dictionaries for the frame
        self.classifier = trained_svm
        self.bounding_box = self.detection['bb']
        self.image_index = 0

    def get_image(self, index):
        img_file = '/home/rohit497/16-720/Project/MultiObjectTracking/2DMOT2015/train/TUD-Stadtmitte/img1/' + images[index]
        img = cv2.imread(img_file)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def active_state(self, all_gts):
        best_obj = -1
        best_overlap = 0
        for i in range(len(all_gts)):
            dx = min(self.detection['bb'][2], all_gts[i]['bb'][2]) - max(self.detection['bb'][0], all_gts[i]['bb'][0])
            dy = min(self.detection['bb'][2], all_gts[i]['bb'][2]) - max(self.detection['bb'][1], all_gts[i]['bb'][1])

            if dx > 0 and dy > 0:
                overlap = dx * dy
            else:
                overlap = 0

            if best_overlap < overlap:
                best_overlap = overlap
                best_obj = all_gts[i]['id']
        self.obj_id = best_obj
        if self.obj_id == -1:
            self.state_type = 'inactive'
        else:
            self.state_type = 'tracked'

    def tracked_state(self, all_detections, all_gts):
        
        x0 = self.bounding_box[0]
        x1 = self.bounding_box[2]
        y0 = self.bounding_box[1]
        y1 = self.bounding_box[3]

        curr_img = self.get_image(self.image_index)

        #cv2.getRectSubPix(curr_img, (x1 - x0, y1 - y0), ((x1 - x0)/2, (y1-y0)/2))

        #template = curr_img[int(y0):int(y1), int(x0):int(x1)]

        lk_params = dict( winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

        feature_params = dict(maxCorners = 3000,
            qualityLevel = 0.5,
            minDistance = 3,
            blockSize = 3)

        pts = np.array([[x0,x0], [x0, y1], [x1, y0], [x1, y1]])
        for x in np.linspace(x0, x1, num=10):
            for y in np.linspace(y0, y1, num=10):
                pts = np.vstack((pts, np.array([x,y])))

        pts = pts.reshape((-1, 1, 2))

        p0 = np.float32(pts).reshape(-1, 1, 2)
        new_img = self.get_image(self.image_index + 1)
        
        # perform LK tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(curr_img, new_img, p0, None, **lk_params)
        
        # for forward-backward
        p0r, st, err = cv2.calcOpticalFlowPyrLK(new_img, curr_img, p1, None, **lk_params)

        p1 = p1.reshape(-1, 2)
        p1_min = np.min(p1, axis=0)
        p1_max = np.max(p1, axis=0)
        bounding_box = [p1_min[0], p1_min[1], p1_max[0], p1_max[1]]

        # get forward-backwards errors
        errors = []
        for i in range(len(pts)):
            e = np.linalg.norm(p0[i] - p0r[i])
            errors.append(e)
        e_med = statistics.median(errors)

        # find the overlap among the object detections in the next image
        best_overlap = -1
        best_det = -1
        for i in range(len(all_detections)):
            det = all_detections[i]
            dx = min(bounding_box[2], det['bb'][2]) - max(bounding_box[0], det['bb'][0])
            dy = min(bounding_box[3], det['bb'][3]) - max(bounding_box[1], det['bb'][1])

            if dx > 0 and dy > 0:
                overlap = dx * dy
            else:
                overlap = -1

            if best_overlap < overlap:
                best_overlap = overlap
                best_det = i

        k_overlaps = self.overlaps[-(k-1):] + [best_overlap]
        o_mean = np.mean(k_overlaps)
        
        if o_mean > 5000 and e_med < 0.05:
            self.overlaps.append(best_overlap)
            self.state_type = 'tracked'
        else:
            self.state_type = 'lost'

        self.image_index += 1
        return best_det

    def lost_state_common(self, all_detections):
        x0 = self.bounding_box[0]
        x1 = self.bounding_box[2]
        y0 = self.bounding_box[1]
        y1 = self.bounding_box[3]

        curr_img = self.get_image(self.image_index)

        #template = self.curr_img[y0:y1, x0:x1]

        lk_params = dict( winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

        feature_params = dict(maxCorners = 3000,
            qualityLevel = 0.5,
            minDistance = 3,
            blockSize = 3)

        # get some feature points
        pts = np.array([[x0,x0], [x0, y1], [x1, y0], [x1, y1]])
        for x in np.linspace(x0, x1, num=10):
            for y in np.linspace(y0, y1, num=10):
                pts = np.vstack((pts, np.array([x,y])))

        pts = pts.reshape((-1, 1, 2))


        p0 = np.float32(pts).reshape(-1, 1, 2)
        new_img = self.get_image(self.image_index + 1)
        
        # perform LK tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(curr_img, new_img, p0, None, **lk_params)

        p1 = p1.reshape(-1, 2)
        p1_min = np.min(p1, axis=0)
        p1_max = np.max(p1, axis=0)
        bounding_box = [p1_min[0], p1_min[1], p1_max[0], p1_max[1]]
        
        # for forward-backward
        p0r, st, err = cv2.calcOpticalFlowPyrLK(new_img, curr_img, p1, None, **lk_params)
        fb_error = np.linalg.norm(p0-p0r)

        box_center_x = (bounding_box[2] - bounding_box[0]) / 2
        box_center_y = (bounding_box[3] - bounding_box[1]) / 2
        box_center = np.array([box_center_x, box_center_y])

        old_height = self.bounding_box[3] - self.bounding_box[1]
        new_height = bounding_box[3] - bounding_box[1]

        LK_height_ratio = new_height / old_height

        i = 0
        dets_to_save = []
        all_features = []
        preds = []
        for det in all_detections:
            score = det['conf']

            dx = min(bounding_box[2], det['bb'][2]) - max(bounding_box[0], det['bb'][0])
            dy = min(bounding_box[3], det['bb'][3]) - max(bounding_box[1], det['bb'][1])

            if dx > 0 and dy > 0:
                overlap = dx * dy
            else:
                overlap = 0

            det_center_x = (det['bb'][2] - det['bb'][0]) / 2
            det_center_y = (det['bb'][3] - det['bb'][1]) / 2
            det_center = np.array([det_center_x, det_center_y])
            dist = np.linalg.norm(box_center - det_center)

            det_height = det['bb'][3] - det['bb'][1]
            det_height_ratio = new_height / det_height

            features = np.array([[dist, fb_error, overlap, det_height_ratio, LK_height_ratio]])
            pred = self.classifier.predict(features)

            preds.append(pred)
            all_features.append(features)
            i+= 1

        return (preds, all_features)
        
    def lost_state(self, all_detections):
        preds, features = self.lost_state_common(all_detections)

        preds = np.array(preds)
        max_index = np.argmax(preds)

        if preds[max_index] > 0:
            self.state_type = 'tracked'
            ret_index = max_index
        else:
            self.state_type = 'lost'
            ret_index = -1

        self.image_index += 1
        return ret_index

    def lost_state_train(self, all_detections, gt):
        preds, features = self.lost_state_common(all_detections)
        pred_detection = np.argmax(preds)

        if preds[pred_detection] <= 0:
            pred_detection = -1

        best_overlap = -1
        best_detection = -1

        if gt == None:
            gt_detection = -1
        else:
            for i in range(len(all_detections)):
                det = all_detections[i]

                dx = min(gt['bb'][2], det['bb'][2]) - max(gt['bb'][0], det['bb'][0])
                dy = min(gt['bb'][3], det['bb'][3]) - max(gt['bb'][1], det['bb'][1])

                if dx > 0 and dy > 0:
                    overlap = dx * dy
                else:
                    overlap = -1

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_detection = i

            k_overlaps = self.overlaps[-(k-1):] + [best_overlap]
            o_mean = statistics.mean(k_overlaps)
            if o_mean > 5000:
                self.overlaps.append(best_overlap)
                gt_action = 6
                gt_detection = best_detection
            else:
                gt_action = 5
                gt_detection = -1


        if pred_detection != gt_detection:
            for i in range(len(all_detections)):
                X = np.array(features[i])
                if gt_detection == i:
                    y = np.array([1])
                else:
                    y = np.array([-1])

                self.classifier.train_on_batch(X, y)

        self.image_index += 1
        return gt_detection

def create_classifier():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_shape=(5,)))
    model.add(keras.layers.Dense(64, activation='sigmoid'))
    model.add(keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train(gt_fname, det_fname):
    MDP_dict = {}
    gts = load_obj(gt_fname)
    dets = load_obj(det_fname)

    classifier = create_classifier()

    # Create new MDP's for every detection in the first frame
    first_gt = gts[1]
    first_det = dets[1]
    for det in first_det:
        m = MDP(det, classifier, gts)
        m.active_state(first_gt)
        if m.obj_id != -1:
            MDP_dict[m.obj_id] = m

    im = get_image(0)

    for det in first_det:
        cv2.rectangle(im,(int(det['bb'][0]),int(det['bb'][1])),(int(det['bb'][2]),int(det['bb'][3])),(0,255,0),1)
    cv2.imshow("image", im)
    cv2.waitKey(0)



    for i in range(2, len(gts)+1):
        print('Iter ' + str(i))
        curr_dets = copy.deepcopy(dets[i]).tolist()
        curr_gts = gts[i]
        seen = {}

        im = get_image(i-1)

        for key in MDP_dict:
            if MDP_dict[key].state_type == 'tracked':
                det = m.tracked_state(curr_dets, curr_gts)
                if det != -1:
                    cv2.rectangle(im,(int(curr_dets[det]['bb'][0]),int(curr_dets[det]['bb'][1])),(int(curr_dets[det]['bb'][2]),int(curr_dets[det]['bb'][3])),(0,255,0),1)
                    curr_dets = curr_dets[:det] + curr_dets[det+1:]
            elif MDP_dict[key].state_type == 'lost':
                imp_gt = None
                for j in range(len(curr_gts)):
                    if key == curr_gts[j]['id']:
                        imp_gt = curr_gts[j]
                        break
                det = m.lost_state_train(curr_dets, imp_gt)
                if det != -1:
                    curr_dets = curr_dets[:det] + curr_dets[det+1:]

        for elem in curr_dets:
            print('\tNew elem!')
            m = MDP(elem, classifier, gts)
            m.active_state(first_gt)
            if m.obj_id != -1 and m.obj_id not in MDP_dict:
                MDP_dict[m.obj_id] = m
                cv2.rectangle(im,(int(elem['bb'][0]),int(elem[det]['bb'][1])),(int(elem['bb'][2]),int(elem['bb'][3])),(0,255,0),1)

        cv2.imshow('iage', im)
        cv2.waitKey(0)

def test(gt_fname, det_fname):
    MDP_dict = {}
    gts = load_obj(gt_fname)
    dets = load_obj(det_fname)

    # Create new MDP's for every detection in the first frame
    first_gt = gts[1]
    first_det = dets[1]
    for det in first_det:
        m = MDP(det)
        m.active_state(first_gt)
        if m.obj_id != -1:
            MDP_dict[m.obj_id] = m

    for i in range(2, len(gts)+1):
        curr_dets = copy.deepcopy(dets[i])
        curr_gts = gts[i]
        seen = {}
        for key in MDP_dict:
            if MDP_dict[key].state_type == 'tracked':
                det = m.tracked_state(curr_dets, curr_gts)
                if det != None:
                    curr_dets.remove(det)
                seen[key] = 0

        for key in MDP_dict:
            if MDP_dict[key].state_type == 'lost' and key not in seen:
                det = m.lost_state(curr_dets)
                if det != None:
                    curr_dets.remove(det)
                seen[key] = 0

        for elem in curr_dets:
            m = MDP(det)
            m.active_state(first_gt)
            if m.obj_id != -1 and m.obj_id not in MDP_dict:
                MDP_dict[m.obj_id] = m


train('/home/rohit497/16-720/Project/MultiObjectTracking/2DMOT2015/train/TUD-Stadtmitte/gt/gt.txt', '/home/rohit497/16-720/Project/MultiObjectTracking/2DMOT2015/train/TUD-Stadtmitte/det/det.txt')






