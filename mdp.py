import cv2

e_threshold = 0.1
o_threshold = 10

class MDP(object):
    def __init__(self, image_prefix, first_image_index, image_suffix, detection):
        self.state_type = 'active'
        self.LK_tracker = None
        self.bounding_box = detection['box_points']
        self.image_prefix = image_prefix
        self.image_index = first_image_index
        self.image_suffix = image_suffix
        self.detection = detection
        self.overlaps = []

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

        feature_params = dict(maxCorners = 3000,
            qualityLevel = 0.5,
            minDistance = 3,
            blockSize = 3)

        pts = cv2.goodFeaturesToTrack(template, **feature_params)
        for i in range(len(pts)):
            # TODO: add back the x0 and y0 offset


        p0 = np.float32(pt).reshape(-1, 1, 2)

        new_img = self.get_image(self.image_index + 1)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(curr_img, new_img, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(new_img, curr_img, p1, None, **lk_params)

        p1 = p1.reshape(-1, 2)
        p1_min = np.min(p1, axis=0)
        p1_max = np.max(p1, axis=0)

        bounding_box = [p1_min[0], p1_min[1], p1_max[0], p1_max[1]]

        errors = []
        for i in range(len(pts)):
            e = np.linalg.norm(p0[i] - p0r[i])
            errors.append(e)
        e_med = statistics.median(errors)

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


