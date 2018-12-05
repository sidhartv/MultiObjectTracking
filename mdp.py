class MDP(object):
    def __init__(self):
        self.state_type = 'active'
        self.object_detector = None
        self.LK_tracker = None
        self.bounding_box = [0,0,0,0]

    def action_reward(self, action, detections):
        pass

    def tracked_reward(self, action):
        pass

    def lost_reward(self, action):
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


