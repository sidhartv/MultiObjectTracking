import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
<<<<<<< HEAD
from parse_obj_xml.py import load_obj
import copy
=======
>>>>>>> edff65027455228a0796a6c0d5677689e5ac4bad

e_threshold = 0.1
o_threshold = 10
k = 4


class MDP(object):
    def __init__(self, image_prefix, first_image_index, image_suffix, detection, trained_svm):
        self.state_type = 'active'
        self.LK_tracker = None
        self.image_prefix = image_prefix
        self.image_index = first_image_index
        self.image_suffix = image_suffix
        self.detection = detection
        self.overlaps = []
        self.gts = gts # list of dictionaries for the frame
        self.classifier = trained_svm

    def get_image(self, index):
        img_file = self.image_prefix + str(index) + self.image_suffix
        img = cv2.imread(img_file)
        return cv2.cvtColor(img, cv2.BGR2GRAY)

    def active_state(self, all_gts):
        best_obj = -1
        best_overlap = 0
        for i in xrange(len(gts)):
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

        template = self.curr_img[y0:y1, x0:x1]

        lk_params = dict( winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

        feature_params = dict(maxCorners = 3000,
            qualityLevel = 0.5,
            minDistance = 3,
            blockSize = 3)

        # get some feature points
        pts = cv2.goodFeaturesToTrack(template, **feature_params)

        # add corners
        pts.append(np.array([[0, 0]]))
        pts.append(np.array([[0, y1-y0]]))
        pts.append(np.array([[x1-x0, 0]]))
        pts.append(np.array([[x1-x0, y1-y0]]))

        # Add the (x0, y0) offset back, because good features were found on the 
        # template image
        for i in range(len(pts)):
            pts[i][0][0] += x0
            pts[i][0][1] += y0


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
        for det in all_detections:
            dx = min(bounding_box[2], det['bb'][2]) - max(bounding_box[0], det['bb'][0])
            dy = min(bounding_box[3], det['bb'][3]) - max(bounding_box[1], det['bb'][1])

            if dx > 0 and dy > 0:
                overlap = dx * dy
            else:
                overlap = 0

            if best_overlap < overlap:
                best_overlap = overlap

        k_overlaps = self.overlaps[-(k-1):] + [best_overlap]
        o_mean = statistics.mean(k_overlaps)
        
        if o_mean > o_threshold and e_med < e_threshold:
            self.overlaps.append(best_overlap)
            self.state_type = 'tracked'
        else:
            self.state_type = 'lost'

    def lost_state_common(self, all_detections):
        x0 = self.bounding_box[0]
        x1 = self.bounding_box[2]
        y0 = self.bounding_box[1]
        y1 = self.bounding_box[3]

        curr_img = self.get_image(self.image_index)

        template = self.curr_img[y0:y1, x0:x1]

        lk_params = dict( winSize  = (10, 10), 
            maxLevel = 5, 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

        feature_params = dict(maxCorners = 3000,
            qualityLevel = 0.5,
            minDistance = 3,
            blockSize = 3)

        # get some feature points
        pts = cv2.goodFeaturesToTrack(template, **feature_params)

        # add corners
        pts.append(np.array([[x0, y0]]))
        pts.append(np.array([[x0, y1]]))
        pts.append(np.array([[x1, y0]]))
        pts.append(np.array([[x1, y1]]))

        # Add the (x0, y0) offset back, because good features were found on the 
        # template image
        for i in range(len(pts)):
            pts[i][0][0] += x0
            pts[i][0][1] += y0


        p0 = np.float32(pt).reshape(-1, 1, 2)
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
        for det in all_detections:
            score = det['percentage_probability']

            dx = min(bounding_box[2], det['box_points'][2]) - max(bounding_box[0], det['box_points'][0])
            dy = min(bounding_box[3], det['box_points'][3]) - max(bounding_box[1], det['box_points'][1])

            if dx > 0 and dy > 0:
                overlap = dx * dy
            else:
                overlap = 0

            det_center_x = (det['box_points'][2] - det['box_points'][0]) / 2
            det_center_y = (det['box_points'][3] - det['box_points'][1]) / 2
            det_center = np.array([det_center_x, det_center_y])
            dist = np.linalg.norm(box_center - det_center)

            det_height = det['box_points'][3] - det['box_points'][1]
            det_height_ratio = new_height / det_height

            features = np.array([dist, fb_error, overlap, det_height_ratio, LK_height_ratio])
            pred = self.classifier.decision_function(features)

            preds.append(pred)
            all_features.append(features)
            i+= 1

        return (preds, features)
        

    def lost_state(self, all_detections):
        preds, features = self.lost_state_common(all_detections)

        preds = np.array(preds)
        max_index = np.argmax(preds)

        if preds[max_index] > 0:
            self.state_type = 'tracked'
        else:
            self.state_type = 'lost'

    def lost_state_train(self, all_detections, gt):
        preds, features = self.lost_state_common(all_detections)
        pred_detection = np.argmax(preds)

        if preds[pred_detection] <= 0:
            pred_detection = -1

        best_overlap = -1
        best_detection = -1

        if ground_truth == None:
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
            if o_mean > o_threshold:
                self.overlaps.append(best_overlap)
                gt_action = 6
                gt_detection = best_detection
            else:
                gt_action = 5
                gt_detection = -1


        if pred_detection != gt_detection:
            for i in range(len(all_detections)):
                X = features[i]
                if gt_detection == i:
                    y = 1
                else:
                    y = -1

                self.classifier.partial_fit(X, y)


def train(gt_fname, det_fname):
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

    for i in xrange(2, len(gts)+1):
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
                imp_gt = None
                for j in xrange(len(curr_gts)):
                    if key == curr_gts[i]['id']:
                        imp_gt = curr_gts[i]['id']
                        break
                det = m.lost_state_train(imp_gt)
                if det != None:
                    curr_dets.remove(det)
                seen[key] = 0

        for elem in curr_dets:
            m = MDP(det)
            m.active_state(first_gt)
            if m.obj_id != -1 and m.obj_id not in MDP_dict:
                MDP_dict[m.obj_id] = m

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

    for i in xrange(2, len(gts)+1):
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

    











