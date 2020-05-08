"""
Once a model is learned, use this to play it.
"""

from game import create_game
import numpy as np
from neural_network import neural_net, LossHistory

NUM_SENSORS = 4


def play(model):
    distance_without_crashing = 0
    max_distance_without_crashing = 0
    car_distance = 0
    game_state = create_game.GameState()
    car_crashed = False
    velocity = []
    distance_until_crash_lst = []

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))

    # Move.
    while True:
        car_distance += 1
        distance_without_crashing += 1
        velocity.append(game_state.velocity_changer)
        average_speed = average(velocity)
        average_distance_until_crash = average(distance_until_crash_lst)
        
        # Choose action.
        action = (np.argmax(model.predict(state, batch_size=1)))

        # Take action. 
        reward, state = game_state.frame_step(action)

        # Update max.
        if distance_without_crashing > max_distance_without_crashing:
            max_distance_without_crashing = distance_without_crashing

        if reward == -1000:
            car_crashed = True
            if distance_without_crashing > 20:
                distance_until_crash_lst.append(distance_without_crashing)

            # Reset.
            distance_without_crashing = 1

        # Prefend error in print statement below.
        if average_distance_until_crash is None:
            average_distance_until_crash = 1

        # Tell us something.     
        if car_distance % 100 == 0 and car_crashed is False: 
            distance_first_run_txt = "The agent hasn't crashed yet, current distance {a}.\tAverage speed {b}."   
            print(distance_first_run_txt.format(a = distance_without_crashing, b = int(average_speed)))
        elif car_distance % 100 == 0 and car_crashed is True:
            max_distance_and_distance_txt = "Current distance {a}.\tAverage speed {b}.\tMax distance without crashing {c}.\tAvg distance {d}."
            print(max_distance_and_distance_txt.format(a = distance_without_crashing, b = int(average_speed), c = max_distance_without_crashing, d = int(average_distance_until_crash)))  
        
def average(velocity):
    if len(velocity) > 1:
        return sum(velocity) / len(velocity)

def average(distance_until_crash_lst):
    if len(distance_until_crash_lst) > 1:
        return sum(distance_until_crash_lst) / len(distance_until_crash_lst) 

if __name__ == "__main__":
    saved_model = 'saved-models/128-128-64-50000-500000.h5' 
    model = neural_net(NUM_SENSORS, [128, 128], saved_model)
    play(model)
