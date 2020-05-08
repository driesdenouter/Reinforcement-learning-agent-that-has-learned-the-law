import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors slows things down.
show_sensors = False

class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(100, 100, 0.5)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)
    
        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(self.create_obstacle(200, 350, 100))
        self.obstacles.append(self.create_obstacle(700, 200, 125))
        self.obstacles.append(self.create_obstacle(600, 600, 35))

        # Create a cat.
        self.create_cat()


    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body


    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)
    
	
    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        self.velocity_changer = 40
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse(driving_direction)
        self.space.add(self.car_body, self.car_shape)

        # Define the possible actions for agent to take.
        self.action_memory = {
            0: [0,0],
            1: [0,1],
            2: [1,0],
            3: [1,1],
            4: [2,0],
            5: [2,1],
        }

    
    def frame_step(self, action):
        
        current_action = self.action_memory[action]
        angle = current_action[0]
        speed = current_action[1]

        minN = 3 # Let speed never get over or under a specific value.
        maxN = 50

        self.velocity_changer = max(minN, self.velocity_changer)
        self.velocity_changer = min(maxN, self.velocity_changer)

        if angle == 0:  # Turn left.
            self.car_body.angle -= .2
        elif angle == 1:  # Turn right.
            self.car_body.angle += .2

        if speed == 0:  # Slow down.
            self.velocity_changer -= 1
        elif speed == 1:  # Speed up.
            self.velocity_changer += 1

        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()

        # Move cat.
        if self.num_steps % 5 == 0:
            self.move_cat() 

        # Speed agent.
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = (50 + ((self.velocity_changer) * 0.005)) * driving_direction 

        # Draw screen slows down the training, so only draw screen in final frames training and playing.
        if self.num_steps < 1490000:
        	draw_screen = False
        	# Update the screen and stuff.
        	screen.fill(THECOLORS["black"])
        	draw(screen, self.space)
        	self.space.step(1./10)
        	if draw_screen:
        		pygame.display.flip()
        		clock.tick()

        else: 
        	draw_screen = True
	        # Update the screen and stuff.
	        screen.fill(THECOLORS["black"])
	        draw(screen, self.space)
	        self.space.step(1./10)
	        if draw_screen:
	            pygame.display.flip()
	            clock.tick()
	        for evt in pygame.event.get():
	            if evt.type == pygame.QUIT:
	                pygame.quit()
	                sys.exit()
              
        
        # Get the current location and the readings of sonar arms and velocity as state.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        normalized_readings = [(x-20.0)/20.0 for x in readings] 
        state = np.array([normalized_readings])

        # Set the reward 
        if self.car_is_crashed(readings):
            # Car crashed when any reading of sonar arms == 1.
            self.crashed = True
            reward = -1000
            self.recover_from_crash(driving_direction)
        elif self.speed_is_violated(): 
        # Set low reward if the speedlimit is violated.
            coef_velo_change = 1.3
            reward = -50 - int(self.velocity_changer ** coef_velo_change)
            self.num_steps += 1
        elif self.speed_within_limits():   
        # Reward is based on the readings (lower reading is better) and the velocity  coefficient (higher velocity is better).
            intercept_reward = -5
            coef_velo_change = 1.738495
            coef_sum_readings = 1.393518
            
            reward = intercept_reward + int(self.velocity_changer ** coef_velo_change) + int(self.sum_readings(readings[0:3]) ** coef_sum_readings)
            self.num_steps += 1

        return reward, state


    def speed_within_limits(self):
        # Set the allowed speed.
        if self.velocity_changer in range(0, 31):
            return True
        else:
            return False


    def speed_is_violated(self):
        # Set the speed limit.
        speed_limit = 30
        if self.velocity_changer > speed_limit:
            return True
        else:
            return False

      
    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction
    
    
    def move_cat(self):
        #randomly move cat.
        speed = random.randint(20, 75)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction
    

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False


    def recover_from_crash(self, driving_direction):
        # We hit something, so recover.
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.


    def sum_readings(self, readings):
        # Sum the number of non-zero readings. 
        tot = 0
        for i in readings:
            tot += i
        return tot


    def get_sonar_readings(self, x, y, angle):
        readings = []
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))
        readings.append(self.velocity_changer)

        if show_sensors:
            pygame.display.update()

        return readings


    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i


    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points


    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)


    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
