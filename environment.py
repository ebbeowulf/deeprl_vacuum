import numpy as np
import cv2
import pdb

FULL_ROOM_SIZE = 50
ROBOT_RADIUS = 2
ROBOT_SIZE=2*ROBOT_RADIUS+1

OBSTACLE=0
EMPTY_SPACE=1
OBJECT_1 = 3
OBJECT_2 = 5

def dist2D(x,y):
    return np.sqrt(x**2+y**2)

def deg2XY(deg):
    x=0
    y=0
    if deg>0 and deg<180:
        y=1
    elif deg<0:
        y=-1
    if deg<90 and deg>-90:
        x = 1
    elif deg>90 or deg<-90:
        x = -1
    return x,y

def deg2rad(deg):
    return deg*np.pi/180.0

class dirty_room:
    def __init__(self, row_range, col_range, solid_obj_maxD, history_size=1000, max_num_steps=500):
        self.row_range = row_range
        self.col_range = col_range
        self.map = np.zeros((FULL_ROOM_SIZE, FULL_ROOM_SIZE))
        if self.row_range[0]<ROBOT_SIZE or self.col_range[0]<ROBOT_SIZE:
            print("Cannot specify a row or col range smaller than the size of the robot")
            exit(-1)
        self.action_space = [0, 1, 2, 3]
        self.solid_obj_maxD = solid_obj_maxD
        self.reward_history = []
        self.reward_history_size = history_size
        self.max_num_steps=max_num_steps

    def update_reward_history(self, rew):
        self.reward_history.append(rew)
        if len(self.reward_history) > self.reward_history_size:
            self.reward_history.pop(0)

    def isInMap(self, row, col):
        if row>=0 and row<FULL_ROOM_SIZE and col>=0 and col<FULL_ROOM_SIZE:
            return True
        return False

    def isInRoom(self, row, col):
        if row>=0 and row<self.dim[0] and col>=0 and col<self.dim[1]:
            return True
        return False

    def reset(self, num_solid_objects):
        D = np.random.random(2)
        self.dim = (int((self.row_range[1]-self.row_range[0])*D[0] + self.row_range[0]),
                    int((self.col_range[1] - self.col_range[0]) * D[1] + self.col_range[0]))
        self.map = np.zeros((FULL_ROOM_SIZE, FULL_ROOM_SIZE), dtype=int)
        self.map[0:self.dim[0],0:self.dim[1]]=np.ones(self.dim, dtype=int)
        # Robot Pose is [pixel, pixel, degrees (should always be an increment of 45)
        self.robot_pose = [int(self.dim[0]/2), int(self.dim[1]/2), 0]

        # print("New Map Dimensions")
        # print(self.dim)
        for i in range(num_solid_objects):
            if np.random.random(1)[0]>0.5:
                self.add_round_object(OBSTACLE)
            else:
                self.add_rect_object(OBSTACLE)

        self.total_empty = np.count_nonzero(self.map)
        self.total_visited = 0
        self.set_visited()
        self.num_steps = 0
        self.episode_rewards = []
        self.observation_space = self.display()
        return self.observation_space

    def check_room_with_object(self, proposed_map):
        # Copy the map
        mp = proposed_map.astype(np.uint8)*255

        # Run erosion to expand the obstacle map - need to feed it an odd-sized kernel
        if ROBOT_RADIUS % 2 == 0:
            kernel = np.ones((ROBOT_RADIUS+1, ROBOT_RADIUS+1), np.uint8)
        else:
            kernel = np.ones((ROBOT_RADIUS, ROBOT_RADIUS), np.uint8)
        mp_erosion = cv2.erode(mp, kernel, iterations=1)

        # Is the robot pose still free?
        if mp_erosion[self.robot_pose[0],self.robot_pose[1]]<100:
            return False

        # Run floodFill over the remaining empty space
        mp2 = mp_erosion.copy()
        mask = np.zeros((FULL_ROOM_SIZE+2, FULL_ROOM_SIZE+2),dtype=np.uint8)
        cv2.floodFill(mp2, mask, (self.robot_pose[1], self.robot_pose[0]), 100)

        # Count reachable cells - allow for a 1% drop
        r1 = np.count_nonzero(mp_erosion>200)
        r2 = np.count_nonzero(mp2==100)
        if (abs(r1-r2))/r1 <0.01:
            return True

        return False

    def add_round_object(self, value):
        countInValid = 0
        # Cannot cover the robot pose
        while countInValid<50:
            countInValid+=1
            D = np.random.random(3)
            obj_pose = [int(self.dim[0]*D[0]), int(self.dim[1]*D[1])]
            obj_rad = self.solid_obj_maxD[0]*D[2]
            if dist2D(obj_pose[0]-self.robot_pose[0], obj_pose[1]-self.robot_pose[1])-obj_rad<=ROBOT_SIZE:
                continue

            mR = int(np.ceil(obj_rad))
            new_map = self.map.copy()
            for x in range(-mR,mR,1):
                for y in range(-mR, mR, 1):
                    if self.isInRoom(obj_pose[0]+x, obj_pose[1]+y) and dist2D(x,y)<obj_rad and new_map[obj_pose[0]+x, obj_pose[1]+y]==EMPTY_SPACE:
                        new_map[obj_pose[0]+x, obj_pose[1]+y] = value

            if value>OBSTACLE or self.check_room_with_object(new_map):
                self.map = new_map
                return True

        print('no round object added')
        return False

    def add_rect_object(self, value):
        countInValid = 0
        # Cannot cover the
        while countInValid < 50:
            countInValid += 1
            D = np.random.random(4)
            obj_pose = [int(self.dim[0] * D[0]), int(self.dim[1] * D[1])]
            obj_dim = [int(self.solid_obj_maxD[0] * D[2]),
                       int(self.solid_obj_maxD[1] * D[3])]
            # Dim check - make sure it doesn't cover the robot pose
            if (obj_pose[0]+obj_dim[0]+ROBOT_SIZE)< self.robot_pose[0] or \
                    (obj_pose[0] - obj_dim[0] - ROBOT_SIZE) > self.robot_pose[0] or \
                    (obj_pose[1] + obj_dim[1] + ROBOT_SIZE) < self.robot_pose[1] or \
                    (obj_pose[1] - obj_dim[1] - ROBOT_SIZE) > self.robot_pose[1]:
                new_map = self.map.copy()
                for x in range(-obj_dim[0], obj_dim[0], 1):
                    for y in range(-obj_dim[1], obj_dim[1], 1):
                        if self.isInRoom(obj_pose[0] + x, obj_pose[1] + y):
                            new_map[obj_pose[0] + x, obj_pose[1] + y] = value

                if self.check_room_with_object(new_map):
                    self.map = new_map
                    return True
        print('no rect object added')
        return False

    def set_visited(self):
        # Try the simplest approach first... the robot might not be large enough
        # to require a better method
        count_new = 0
        dx,dy = deg2XY(self.robot_pose[2])
        pList = []
        for r in np.arange(-ROBOT_RADIUS, ROBOT_RADIUS,0.2):
            th = deg2rad(self.robot_pose[2])+np.pi/2

            y = r*np.sin(th)
            x = r*np.cos(th)
            # Add the single row in the direction
            yi = int(self.robot_pose[0]+y)
            xi = int(self.robot_pose[1]+x)
            pList.append([yi,xi])
            if dy!=0:
                pList.append([yi-dy, xi])
            else:
                pList.append([yi, xi-dx])

        for pt in pList:
            if self.map[pt[0],pt[1]]==EMPTY_SPACE:
                count_new += 1
                self.map[pt[0],pt[1]] = EMPTY_SPACE+1
            elif self.map[pt[0],pt[1]]==OBJECT_1:
                count_new += 1
                self.map[pt[0],pt[1]] = OBJECT_1+1
            elif self.map[pt[0], pt[1]] == OBJECT_2:
                count_new += 1
                self.map[pt[0], pt[1]] = OBJECT_2 + 1
        self.total_visited += count_new
        return count_new

    def isValidRobotPose(self, newR, newC):
        for row in range(-ROBOT_RADIUS, ROBOT_RADIUS+1):
            for col in range(-ROBOT_RADIUS, ROBOT_RADIUS+1):
                if dist2D(row,col)<=ROBOT_RADIUS:
                    if not self.isInRoom(newR+row, newC+col) or self.map[newR+row, newC+col]<1:
                        return False
        return True

    def move_robot(self, action):
        #Need to do something complicated to model the "bump"
        #   So find the target direction to move the robot. Should be one of 8 values, including diagonals
        #   If the direction targeted is blocked, then try the value to the right, then the value to the left
        #   The robot will move to either of those directions if not blocked, but without rotating.
        self.last_pose = self.robot_pose
        if action==2: #rotate left
            self.robot_pose[2]+=45
            if self.robot_pose[2]>180:
                self.robot_pose[2]-=360
            # print(self.robot_pose)
            return self.set_visited()
        elif action==3: #rotate right
            self.robot_pose[2] -= 45
            if self.robot_pose[2]<=-180:
                self.robot_pose[2] += 360
            # print(self.robot_pose)
            return self.set_visited()
        elif action==0: # move forward
            mult = 1
        elif action==1: # move backward
            mult = -1

        x, y = deg2XY(self.robot_pose[2])
        new_pose = [self.robot_pose[0]+mult*y, self.robot_pose[1]+mult*x, self.robot_pose[2]]
        if self.isValidRobotPose(new_pose[0], new_pose[1]):
            self.robot_pose = new_pose
            return self.set_visited()

        # Try to the right
        newTh = self.robot_pose[2]-45
        if newTh<=-180:
            newTh += 360
        x, y = deg2XY(newTh)
        new_pose = [self.robot_pose[0]+mult*y, self.robot_pose[1]+mult*x, self.robot_pose[2]]
        if self.isValidRobotPose(new_pose[0], new_pose[1]):
            self.robot_pose = new_pose
            return self.set_visited()

        # Try to the left
        newTh = self.robot_pose[2]+45
        if newTh>180:
            newTh -= 360
        x, y = deg2XY(newTh)
        new_pose = [self.robot_pose[0]+mult*y, self.robot_pose[1]+mult*x, self.robot_pose[2]]
        if self.isValidRobotPose(new_pose[0], new_pose[1]):
            self.robot_pose = new_pose
            return self.set_visited()

        # Failed
        return 0

    def display(self):
        img = np.zeros((self.map.shape[0],self.map.shape[1],3), dtype=np.uint8)
        # draw the map
        for y in range(self.dim[0]):
            for x in range(self.dim[1]):
                if self.map[y,x]==EMPTY_SPACE: # Empty Space
                    img[y,x] = 255
                elif self.map[y, x] == (EMPTY_SPACE+1):  # Visited Empty Space
                    img[y, x] = [200, 200, 250]
                elif self.map[y, x] == OBJECT_1:       #Object 1
                    img[y, x] = [100, 200, 100]
                elif self.map[y, x] == (OBJECT_1+1):  # Visited Object 1
                    img[y, x] = [0, 200, 250]
                elif self.map[y, x] == OBJECT_2:  # Object 2
                    img[y, x] = [250, 100, 100]
                elif self.map[y, x] == (OBJECT_2 + 1):  # Visited Object 2
                    img[y, x] = [250, 100, 250]

        # draw the robot
        for y in range(-ROBOT_RADIUS, ROBOT_RADIUS):
            for x in range(-ROBOT_RADIUS, ROBOT_RADIUS):
                if dist2D(y,x)>ROBOT_RADIUS:
                    continue
                theta = np.arctan2(y,x)-deg2rad(self.robot_pose[2])
                if abs(theta)>0.5:
                    img[self.robot_pose[0]+y, self.robot_pose[1]+x] = [0,0,255]
                else:
                    img[self.robot_pose[0] + y, self.robot_pose[1] + x] = [0, 0, 0]
        return img

    def get_reward(self, countNew, action):
        rew = -0.01
        done = False
        # Success
        if (self.total_visited/self.total_empty)>0.95:
            rew = 50
            done = True
        # Failure
        elif self.num_steps>self.max_num_steps:
            rew = -10*(1-self.total_visited/self.total_empty)
            done = True
        elif countNew > 2:
            rew = 0.01
        elif countNew==0:
            rew = -0.05
        return rew, done

    def act(self, action):
        self.num_steps += 1
        countNew = self.move_robot(action)
        reward, done = self.get_reward(countNew, action)
        outI = self.display()
        # Track reward per game
        self.episode_rewards.append(reward)
        # Track reward across games
        self.update_reward_history(reward)
        return outI, reward, done

class semantic_room(dirty_room):
    def __init__(self, row_range, col_range, clutter_noise=0.3, one_way_stuck_pct=0.1, history_size=5000, max_num_steps=500):
        solid_obj_maxD = [20, 20]
        self.clutter_noise = clutter_noise
        self.one_way_stuck_pct = one_way_stuck_pct
        super().__init__(row_range, col_range, solid_obj_maxD, history_size, max_num_steps)

    def reset(self, num_solid_objects):
        D = np.random.random(2)
        self.dim = (int((self.row_range[1]-self.row_range[0])*D[0] + self.row_range[0]),
                    int((self.col_range[1] - self.col_range[0]) * D[1] + self.col_range[0]))
        self.map = np.zeros((FULL_ROOM_SIZE, FULL_ROOM_SIZE), dtype=int)
        self.map[0:self.dim[0],0:self.dim[1]]=np.ones(self.dim, dtype=int)
        # Robot Pose is [pixel, pixel, degrees (should always be an increment of 45)
        self.robot_pose = [int(self.dim[0]/2), int(self.dim[1]/2), 0]

        # print("New Map Dimensions")
        # print(self.dim)
        self.one_way_obj = None
        # self.add_oneway_rect(OBJECT_2)
        # self.add_round_object(OBJECT_1) # add second so that it can overlap with the couch
        self.add_rect_object(OBSTACLE)

        self.total_empty = np.count_nonzero(self.map)-np.count_nonzero(self.map==OBJECT_1)
        self.total_visited = 0
        self.set_visited()
        self.num_steps = 0
        self.episode_rewards = []
        self.gt_map = np.copy(self.map)

        # For every clutter cell, there is a chance that the robot incorrectly classifies it
        #   as clutter
        for row in range(self.dim[0]):
            for col in range(self.dim[1]):
                if self.map[row,col]==OBJECT_1:  #Do this for clutter
                    if np.random.random(1)[0]<self.clutter_noise:
                        self.map[row][col]=EMPTY_SPACE # can't see clutter at this location

        # Now add noise
        self.observation_space = self.display()
        return self.observation_space

    def add_oneway_rect(self, value):
        countInValid = 0
        max_obj_dim = [20,10]
        half_dim = [10,5]
        # Need to pick a direction with adequate space on the edges for the
        #   robot to travel through empty space
        if self.dim[0]>(max_obj_dim[0]+ROBOT_SIZE):
            if self.dim[1]>(max_obj_dim[0]+ROBOT_SIZE) and np.random.random(1)[0]>0.5:
                dir = 1
            else:
                dir = 0
        elif self.dim[1]>(max_obj_dim[0]+ROBOT_SIZE):
            dir = 1
        else:
            print('no one-way object added')
            self.one_way_obj = None
            return False

        while countInValid < 50:
            countInValid += 1
            D = np.random.random(2)
            if dir==0: # object runs along vertical
                obj_pose = [1+ROBOT_RADIUS + half_dim[0] + int((self.dim[0]-max_obj_dim[0]-ROBOT_SIZE) * D[0]),
                            half_dim[1] + int((self.dim[1]-max_obj_dim[1]) * D[1])]
                obj_dim = half_dim
            else:
                obj_pose = [half_dim[1] + int((self.dim[0]-max_obj_dim[1]) * D[0]),
                            1+ROBOT_RADIUS + half_dim[0] + int((self.dim[1]-max_obj_dim[0]-ROBOT_SIZE) * D[1])]
                obj_dim = [half_dim[1],half_dim[0]]

            new_map = np.copy(self.map)
            for x in range(-obj_dim[0], obj_dim[0], 1):
                for y in range(-obj_dim[1], obj_dim[1], 1):
                    if self.isInRoom(obj_pose[0] + x, obj_pose[1] + y):
                        new_map[obj_pose[0] + x, obj_pose[1] + y] = value
            #
            #     if self.check_room_with_object(new_map):
            self.map = new_map
            self.one_way_obj = {'pose': obj_pose, 'dim': obj_dim, 'dir': dir}
            return True

        print('no rect object added')
        self.one_way_obj = None
        return False

    def get_reward(self, countNew, action):
        rew = 0.0
        if countNew==0:
            rew = -0.05
        elif action<2:
            rew = 0.01*countNew-0.01
        done = False
        # Success
        if (self.total_visited/self.total_empty)>0.90:
            rew = 50
            done = True
        # Failure
        elif self.num_steps>self.max_num_steps:
            rew = -10*(1-self.total_visited/self.total_empty)
            done = True
        # elif countNew==0:
        #     rew = -0.05

        # Check clutter condition
        for row in range(self.robot_pose[0]-ROBOT_RADIUS,self.robot_pose[0]+ROBOT_RADIUS+1,1):
            for col in range(self.robot_pose[1] - ROBOT_RADIUS, self.robot_pose[1] + ROBOT_RADIUS+1, 1):
                if self.map[row,col]==(OBJECT_1+1) or (self.map[row,col]==(EMPTY_SPACE+1) and self.gt_map[row,col]==OBJECT_1):
                    # The robot is stuck... stop
                    return -50, True

        # Check one-way couch condition
        # if self.one_way_obj:
        #     if self.gt_map[self.robot_pose[0],self.robot_pose[1]]==OBJECT_2:
        #         #The robot is under the 3D object
        #         #Is it moving in the wrong direction?
        #         isWrong = False
        #         if self.one_way_obj['dir']==0 and self.robot_pose[1]!=self.last_pose[1]:
        #             isWrong = True
        #         if self.one_way_obj['dir']==1 and self.robot_pose[0]!=self.last_pose[0]:
        #             isWrong = True
        #         # If it is not traveling in a straight line under the one-way object
        #         #   Then check to see if the robot has become stuck (True with some percentage)
        #         if isWrong and np.random.random(1)[0]<self.one_way_stuck_pct:
        #             return -10, True
        #         # elif rew>0:
        #         #     rew*=2  # provide additional reward for vacuuming under the object?

        return rew, done
import readchar

def main(args=None):
    env = dirty_room([30, FULL_ROOM_SIZE], [30, FULL_ROOM_SIZE], [10,30])
    env.reset(3)
    img = env.display()
    outI = cv2.resize(img, (400, 400))
    cv2.imshow('coverage_map', outI)
    cv2.waitKey(1)
    while True:
        img = env.display()
        outI = cv2.resize(img,(400,400))
        cv2.imshow('coverage_map',outI)
        cv2.waitKey(1)
        k = readchar.readchar()
        if k == 'w':
            action = 0
        elif k=='s':
            action = 1
        elif k=='a':
            action = 2
        elif k=='d':
            action = 3
        elif k=='p':
            action = 0
            pdb.set_trace()
        elif k=='Q':
            exit(-1)
        env.move_robot(action)

if __name__ == '__main__':
    main()
