import math

class Skeleton:
    def __init__(self, time=0, shoulder_r=(0,0,0), shoulder_l=(0,0,0), elbow_r=(0,0,0), 
                 elbow_l=(0,0,0), wrist_r=(0,0,0), wrist_l=(0,0,0), hip_r=(0,0,0), hip_l=(0,0,0),
                 knee_r=(0,0,0), knee_l=(0,0,0), ankle_r=(0,0,0), ankle_l=(0,0,0)):

        # Input
        self.time = time

        self.shoulder_r = shoulder_r
        self.shoulder_l = shoulder_l

        self.elbow_r = elbow_r
        self.elbow_l = elbow_l

        self.wrist_r = wrist_r
        self.wrist_l = wrist_l

        self.hip_r = hip_r
        self.hip_l = hip_l

        self.knee_r = knee_r
        self.knee_l = knee_l

        self.ankle_r = ankle_r
        self.ankle_l = ankle_l

        # Created
        # Right shoulder angles
        self.shoulder_elbow_r_originated = (self.elbow_r[0] - self.shoulder_r[0], 
                                            self.elbow_r[1] - self.shoulder_r[1], 
                                            self.elbow_r[2] - self.shoulder_r[2])

        self.shoulder_r_alpha = self.calculate_alpha(self.shoulder_elbow_r_originated)
        self.shoulder_r_beta = self.calculate_beta(self.shoulder_elbow_r_originated)

        # Left shoulder angles
        self.shoulder_elbow_l_originated = (self.elbow_l[0] - self.shoulder_l[0], 
                                            self.elbow_l[1] - self.shoulder_l[1], 
                                            self.elbow_l[2] - self.shoulder_l[2])
        
        self.shoulder_l_alpha = self.calculate_alpha(self.shoulder_elbow_l_originated)
        self.shoulder_l_beta = self.calculate_beta(self.shoulder_elbow_l_originated)

        # Right elbow angles
        self.elbow_wrist_r_originated = (self.wrist_r[0] - self.elbow_r[0],
                                         self.wrist_r[1] - self.elbow_r[1],
                                         self.wrist_r[2] - self.elbow_r[2])
        
        self.elbow_r_alpha = self.calculate_alpha(self.elbow_wrist_r_originated)
        self.elbow_r_beta = self.calculate_beta(self.elbow_wrist_r_originated)

        # Left elbow angles
        self.elbow_wrist_l_originated = (self.wrist_l[0] - self.elbow_l[0],
                                         self.wrist_l[1] - self.elbow_l[1],
                                         self.wrist_l[2] - self.elbow_l[2])
        
        self.elbow_l_alpha = self.calculate_alpha(self.elbow_wrist_l_originated)
        self.elbow_l_beta = self.calculate_beta(self.elbow_wrist_l_originated)

        # Right hip angles
        self.hip_knee_r_originated = (self.knee_r[0] - self.hip_r[0], 
                                      self.knee_r[1] - self.hip_r[1], 
                                      self.knee_r[2] - self.hip_r[2])

        self.hip_r_alpha = self.calculate_alpha(self.hip_knee_r_originated)
        self.hip_r_beta = self.calculate_beta(self.hip_knee_r_originated)

        # Left hip angles
        self.hip_knee_l_originated = (self.knee_l[0] - self.hip_l[0], 
                                      self.knee_l[1] - self.hip_l[1], 
                                      self.knee_l[2] - self.hip_l[2])

        self.hip_l_alpha = self.calculate_alpha(self.hip_knee_l_originated)
        self.hip_l_beta = self.calculate_beta(self.hip_knee_l_originated)

        # Right knee angles
        self.knee_ankle_r_originated = (self.ankle_r[0] - self.knee_r[0], 
                                      self.ankle_r[1] - self.knee_r[1], 
                                      self.ankle_r[2] - self.knee_r[2])

        self.knee_r_alpha = self.calculate_alpha(self.knee_ankle_r_originated)
        self.knee_r_beta = self.calculate_beta(self.knee_ankle_r_originated)

        # Left knee angles
        self.knee_ankle_l_originated = (self.ankle_l[0] - self.knee_l[0], 
                                      self.ankle_l[1] - self.knee_l[1], 
                                      self.ankle_l[2] - self.knee_l[2])

        self.knee_l_alpha = self.calculate_alpha(self.knee_ankle_l_originated)
        self.knee_l_beta = self.calculate_beta(self.knee_ankle_l_originated)

        # Spine stuff
        self.midpoint_shoulder = ((self.shoulder_r[0] + self.shoulder_l[0])/2, (self.shoulder_r[1] + self.shoulder_l[1])/2, (self.shoulder_r[2] + self.shoulder_l[2])/2)

        self.midpoint_hip = ((self.hip_r[0] + self.hip_l[0])/2, (self.hip_r[1] + self.hip_l[1])/2, (self.hip_r[2] + self.hip_l[2])/2)

        self.spine_originated = (self.midpoint_shoulder[0] - self.midpoint_hip[0],
                                 self.midpoint_shoulder[1] - self.midpoint_hip[1],
                                 self.midpoint_shoulder[2] - self.midpoint_hip[2])

        self.spine_alpha = self.calculate_alpha(self.spine_originated)
        self.spine_beta = self.calculate_beta(self.spine_originated)

    def __repr__(self) -> str:
        return f"Skeleton ({self.time})\n\tRight Elbow Alpha = {self.elbow_r_alpha}\n\tRight Elbow Beta = {self.elbow_r_beta}\
            \n\tLeft Elbow Alpha = {self.elbow_l_alpha}\n\tLeft Elbow Beta = {self.elbow_l_beta}\
            \n\tRight Shoulder Alpha = {self.shoulder_r_alpha}\n\tRight Shoulder Beta = {self.shoulder_r_beta}\
            \n\tLeft Shoulder Alpha = {self.shoulder_l_alpha}\n\tLeft Shoulder Beta = {self.shoulder_l_beta}\
            \n\tRight Hip Alpha = {self.hip_r_alpha}\n\tRight Hip beta = {self.hip_r_beta}\
            \n\tLeft Hip Alpha = {self.hip_l_alpha}\n\tLeft Hip Beta = {self.hip_l_beta}\
            \n\tRight Knee Alpha = {self.knee_r_alpha}\n\tRight Knee Beta = {self.knee_r_beta}\
            \n\tLeft Knee Alpha = {self.knee_l_alpha}\n\tLeft Knee Beta = {self.knee_l_beta}\
            \n\tSpine Alpha = {self.spine_alpha}\n\tSpine Beta {self.spine_beta}"
    
    def __sub__(self, prev):
        # print(self)
        # print(prev)
        # print(f"Right Shoulder Alpha diff = {self.shoulder_r_alpha - prev.shoulder_r_alpha}")
        # print(f"Right Shoulder Beta diff = {self.shoulder_r_beta - prev.shoulder_r_beta}")

        # print(f"Left Shoulder Alpha diff = {self.shoulder_l_alpha - prev.shoulder_l_alpha}")
        # print(f"Left Shoulder Beta diff = {self.shoulder_l_beta - prev.shoulder_l_beta}")

        # print(f"Right Elbow Alpha diff = {self.elbow_r_alpha - prev.elbow_r_alpha}")
        # print(f"Right Elbow Beta diff = {self.elbow_r_beta - prev.elbow_r_beta}")

        # print(f"Left Elbow Alpha diff = {self.elbow_l_alpha - prev.elbow_l_alpha}")
        # print(f"Left Elbow Beta diff = {self.elbow_l_beta - prev.elbow_l_beta}")

        # print(f"Right Hip Alpha diff = {self.hip_r_alpha - prev.hip_r_alpha}")
        # print(f"Right Hip Beta diff = {self.hip_r_beta - prev.hip_r_beta}")

        # print(f"Left Hip Alpha diff = {self.hip_l_alpha - prev.hip_l_alpha}")
        # print(f"Left Hip Beta diff = {self.hip_l_beta - prev.hip_l_beta}")

        # print(f"Right Knee Alpha diff = {self.knee_r_alpha - prev.knee_r_alpha}")
        # print(f"Right Knee Beta diff = {self.knee_r_beta - prev.knee_r_beta}")

        # print(f"Left Knee Alpha diff = {self.knee_l_alpha - prev.knee_l_alpha}")
        # print(f"Left Knee Beta diff = {self.knee_l_beta - prev.knee_l_beta}")

        # print(f"Spine Alpha diff = {self.spine_alpha - prev.spine_alpha}")
        # print(f"Spine Beta diff = {self.spine_beta - prev.spine_beta}")

        return [self.shoulder_r_alpha - prev.shoulder_r_alpha, self.shoulder_r_beta - prev.shoulder_r_beta,
                self.shoulder_l_alpha - prev.shoulder_l_alpha, self.shoulder_l_beta - prev.shoulder_l_beta,
                self.elbow_r_alpha - prev.elbow_r_alpha, self.elbow_r_beta - prev.elbow_r_beta,
                self.elbow_l_alpha - prev.elbow_l_alpha, self.elbow_l_beta - prev.elbow_l_beta,
                self.hip_r_alpha - prev.hip_r_alpha, self.hip_r_beta - prev.hip_r_beta,
                self.hip_l_alpha - prev.hip_l_alpha, self.hip_l_beta - prev.hip_l_beta,
                self.knee_r_alpha - prev.knee_r_alpha, self.knee_r_beta - prev.knee_r_beta,
                self.knee_l_alpha - prev.knee_l_alpha, self.knee_l_beta - prev.knee_l_beta,
                self.spine_alpha - prev.spine_alpha, self.spine_beta - prev.spine_beta]
    
    
    # Takes a vector, projects to the horizontal plane, and calculates the angle 
    # between that projection and the x-axis
    def calculate_alpha(self, originated_vector):

        # Define the horizontal projection of the vector
        x3, y3, _ = (originated_vector)
        # Define the origin 
        x2, y2, _ = (0, 0, 0)
        # Define i-hat
        x1, y1, _ = (1, 0, 0)

        # Calculate the angle between the three points
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        # # Check if the angle is less than zero.
        # if angle < 0:

        #     # Add 360 to the found angle.
        #     angle += 360
        
        # Return the calculated angle.
        return angle
    
    # Takes a vector, projects to the vertical plane, and calculates the angle 
    # between that projection and the y-axis
    def calculate_beta(self, originated_vector):
        # Define the horizontal projection of the vector
        _, x3, y3 = (originated_vector)
        # Define the origin 
        _, x2, y2 = (0, 0, 0)
        # Define j-hat
        _, x1, y1 = (0, 1, 0)

        # Calculate the angle between the three points
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        # # Check if the angle is less than zero.
        # if angle < 0:

        #     # Add 360 to the found angle.
        #     angle += 360
        
        # Return the calculated angle.
        return angle
    
    def getAry(self):
        return [self.elbow_r_alpha, self.elbow_r_beta, self.shoulder_r_alpha, self.shoulder_r_beta, self.hip_r_alpha, self.hip_r_beta,
                self.knee_r_alpha, self.knee_r_beta, self.elbow_l_alpha, self.elbow_l_beta, self.shoulder_l_alpha, self.shoulder_l_beta,
                self.hip_l_alpha, self.hip_l_beta, self.knee_l_alpha, self.knee_l_beta, self.spine_beta]

# first = Skeleton(time=0, shoulder_r=(3, 0, 7), elbow_r=(4, 3, 5), 
#                          shoulder_l=(-3, 0, 7), elbow_l=(-4, 3, 5), 
#                          wrist_r=(4, 0, 5.1), wrist_l=(-4, 6, 5),
#                          hip_r=(2, 0, 3), hip_l=(-2, 0, 3),
#                          knee_r=(1.5, 0, 1.5), knee_l=(-1.5, 0, 1.5),
#                          ankle_r=(1, 0, 0), ankle_l=(-1, 0, 0))

# arr = first.getAry()
# print(first)
# print(arr)