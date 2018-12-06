import cv2
import numpy as np
from sklearn import svm

e_threshold = 0.1
o_threshold = 10
k = 4


class MDP(object):
    def __init__(self, image_prefix, first_image_index, image_suffix, detection, trained_svm):
        self.state_type = 'active'
        self.LK_tracker = None
        self.bounding_box = detection['box_points']
        self.image_prefix = image_prefix
        self.image_index = first_image_index
        self.image_suffix = image_suffix
        self.detection = detection
        self.overlaps = []
        self.lost_svm = trained_svm

    def get_image(self, index):
        img_file = self.image_prefix + str(index) + self.image_suffix
        img = cv2.imread(img_file)
        return cv2.cvtColor(img, cv2.BGR2GRAY)

    def active_state(self):
        # threshold the probability
        reward = self.detection['percentage_probability'] - 80.0
        reward = max(reward, -20)
        if reward < 0:
            # inactive
            return (2, reward)
        else:
            # track
            return (1, reward)

    def tracked_state(self, all_detections):
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
            dx = min(bounding_box[2], det['box_points'][2]) - max(bounding_box[0], det['box_points'][0])
            dy = min(bounding_box[3], det['box_points'][3]) - max(bounding_box[1], det['box_points'][1])

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
            return (3, 1)
        else:
            return (4, -1)

    def lost_state(self):
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
            pred = self.lost_svm.predict(features)








        

    def lost_state_train(self, ground_truth):
        pass


        


    def transition(self, action):
        if self.state_type == 'active':
            if action not in [1,2]:
                return False
            elif action == 1:
                self.state_type = 'tracked'
                return active_reward(1)
            else:
                self.state_type = 'inactive'
                return active_reward(2)

        elif self.state_type == 'tracked':
            if action not in [3,4]:
                return False
            elif action == 3:
                self.state_type = 'tracked'
                return tracked_reward(3)
            else:
                self.state_type = 'lost'
                return tracked_reward(4)

        elif self.state_type == 'lost':
            if action not in [5,6,7]:
                return False
            elif action == 5:
                self.state_type = 'lost'
                return lost_reward(5)
            elif action == 6:
                self.state_type = 'tracked'
                return lost_reward(6)
            else:
                self.state_type = 'inactive'
                return lost_reward(7)

        else:
            self.state_type = 'inactive'
            return 0


