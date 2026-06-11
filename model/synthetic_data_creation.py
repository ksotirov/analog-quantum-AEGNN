import numpy as np
import random


###################################################################################################################################################################
######################################################################### Hexagon Dataset #########################################################################
###################################################################################################################################################################


#### Tools to generate the datasets

#Choose the direction
def pick_direction(direction_probabilities: np.ndarray) -> float:

    cum_dir_probabilities = np.cumsum(direction_probabilities)

    random_dir_to_go = random.uniform(0, 1)
    next_node = 0
    for i in range(len(cum_dir_probabilities)):
        if cum_dir_probabilities[i] >= random_dir_to_go:
            next_node = i
            break

    return next_node


def add_time_component(prev_time: float, simple_time: bool = True):

    new_time = 0
    if simple_time:
        new_time = prev_time + 1

    return new_time

#Check whether we have walked on the path
def are_coords_new(new_coord: tuple, previous_coords: dict):

    threshold = 0.1
    for coord in list(previous_coords.values()):
        if np.sqrt((new_coord[0] - coord[0]) ** 2 + (new_coord[1] - coord[1]) ** 2) < threshold:
            return False

    return True

#Condition to check whether the new coordinate is far away from all the nodes in a certain window of events. Useful for the quantum algorithm.
def coords_within_radius(new_coord: tuple, previous_coords: dict, allowed_max_radius: float = 0, window: int = 9):

    memory_window = min(window, len(previous_coords))

    if allowed_max_radius != 0:

        all_previous_coordinates = list(previous_coords.values())
        coordinates_in_window = all_previous_coordinates[-memory_window:]
        coordinates_in_window.append(new_coord)

        coordinates_to_check = np.array([[co[0],co[1]] for co in coordinates_in_window])


        recenter_mean = np.mean(np.vstack((np.max(coordinates_to_check,axis=0), np.min(coordinates_to_check,axis=0))), axis=0)

        coordinates_to_check = coordinates_to_check - recenter_mean # recenter everything around 0


        for coord in coordinates_to_check:
            if np.linalg.norm(coord) > allowed_max_radius:
                return False

    return True


#Add a new node
def create_new_node(previous_coords, current_coordinates,
                    direction, new_point_distance,
                    incorporate_time: bool, incorporate_polarity: bool,
                    allow_repeat: bool,
                    max_radius: float,
                    change_polarity: bool,
                    red_dot_currently: bool):

    #Specify the new coordinates in 2D
    new_x = current_coordinates[0] + new_point_distance*np.cos(direction)
    new_y = current_coordinates[1] + new_point_distance*np.sin(direction)
    new_coord = (new_x, new_y)

    #Add time if required
    if incorporate_time:
        new_t = add_time_component(current_coordinates[2])
        new_coord += (new_t,)

    if incorporate_polarity:
        if change_polarity and red_dot_currently:
            new_coord += (-1,)
        else:
            new_coord += (1,)

    if (are_coords_new(new_coord, previous_coords) and coords_within_radius(new_coord, previous_coords, max_radius)) or allow_repeat:
        previous_coords[len(previous_coords)] = new_coord

    elif incorporate_time: #This only occurs if time is incorporated
        new_coord = (new_x, new_y, current_coordinates[2]) #No new time is required in this case

    return new_coord, previous_coords


#### Generate a single step for the different classes

#This defines the honeycomb step
def class_a_step(current_coordinates: tuple[float, float], previous_coords: dict[int, tuple[float, float]],
                 new_point_distance: float,
                 p: float,
                 left_layout: bool, red_dot_currently: bool,
                 incorporate_time: bool, incorporate_polarity: bool,
                 allow_repeat: bool,
                 max_radius: float,
                 change_polarity: bool) -> tuple[tuple[float, float, float, float], list[tuple[float, float, float, float]], bool, bool]:

    direction = 0
    if red_dot_currently:

        possible_direction_prob = np.ones(6)/6                 #Every possible path is equally likely
        next_node = pick_direction(possible_direction_prob)
        direction = 2*np.pi*next_node/6                        #We use Euler's formula as the direction

        left_layout = False
        if (next_node % 2 == 0):
            left_layout = True

        red_dot_currently = False                              #We are definitely not a red point now, since we escaped the middle

    else:
        possible_direction_prob = np.ones(6)
        if left_layout:
            for i in range(1,6,2):
                possible_direction_prob[i] = p                 #Reweight the direction
        else:
            for i in range(0,6,2):
                possible_direction_prob[i] = p                 #Reweight the direction

        #Normalise between 0 and 1
        possible_direction_prob /= np.sum(possible_direction_prob)

        next_node = pick_direction(possible_direction_prob)
        direction = 2*np.pi*next_node/6                        #We use Euler's formula as the direction

        if (left_layout and (next_node % 2 == 1)) or ((left_layout == False) and (next_node % 2 == 0)):
            red_dot_currently = True

        if not red_dot_currently:
            left_layout = not left_layout


    new_coord, previous_coords = create_new_node(previous_coords, current_coordinates, direction, new_point_distance, incorporate_time, incorporate_polarity, allow_repeat, max_radius, change_polarity, red_dot_currently)

    return new_coord, previous_coords, left_layout, red_dot_currently


#This defines the Kagome site step.
def class_b_step(current_coordinates: tuple[float, float], previous_coords: dict[tuple[float, float]],
                 new_point_distance: float,
                 p: float,
                 layout_type: int, red_dot_currently: bool,
                 incorporate_time: bool, incorporate_polarity: bool,
                 allow_repeat: bool,
                 max_radius: float,
                 change_polarity: bool) -> tuple[tuple[float, float, float, float], list[tuple[float, float, float, float]], bool, bool]:


    direction = 0
    if red_dot_currently:

        #Get the overall probability
        possible_direction_prob = np.ones(6)/6

        #Pick the direction
        next_node = pick_direction(possible_direction_prob)
        direction = 2*np.pi*next_node/6

        #Change the state of the probability
        layout_type = next_node % 3

        red_dot_currently = False

    else:

        #Get the overall probability
        possible_direction_prob = np.ones(6)

        #Change the probability of the directions in need
        for i in range(layout_type, 6, 3):
            possible_direction_prob[i] = p

        #Normalise between 0 and 1
        possible_direction_prob /= np.sum(possible_direction_prob)

        #Pick the direction
        next_node = pick_direction(possible_direction_prob)
        direction = 2*np.pi*next_node/6

        #Change the state of the probability
        if (next_node % 3 == layout_type):
            red_dot_currently = True

        if not red_dot_currently:

            #This is based on the following:
            #From 0 pick 1 -> 2
            #From 0 pick 2 -> 1
            #From 1 pick 0 -> 2
            #From 1 pick 2 -> 0
            #From 2 pick 1 -> 0
            #From 2 pick 0 -> 1
            #This can be summerised as:
            #f(0, 1) = 2
            #f(0, 2) = 1
            #f(1, 2) = 0
            #This condition is fulfilled with the property below.
            layout_type = (2*(layout_type + next_node) % 3)

    new_coord, previous_coords = create_new_node(previous_coords, current_coordinates, direction, new_point_distance, incorporate_time, incorporate_polarity, allow_repeat, max_radius, change_polarity, red_dot_currently)

    return new_coord, previous_coords, layout_type, red_dot_currently


#### Generate the graphs

#Constructing the honeycomb graph
# Explanation: The coordinates (x,y) are kept and each point has an associated state with it. This state assings the
# weight of the direction we want to go after the comptutation. Since this is a triangular lattice, we have
# 6 possible directions, which in 2D it is the 6 roots of unity on a complex plane.
# For each coordinate we have 2 different options: either it is a red dot (i.e. outside of the typical bounds)
# or it is a blue dot (i.e. inside the expected region)
# If we are in the red dot, that is, in the center of a hexagon, then we have equal probability of jumping along each node
# If we are in the blue dot, then we have a state to keep in mind. This state can be left
# (i.e. it can jump with weight p=1 in the 0th, 2nd or 4th direction, and with p_0 < p in the 1st, 3rd and 5th direction)
# or right (it can jump with weight p=1 in the 1st, 3rd or 5th direction, and with p_0 < p in the 0th, 2nd or 4th direction).
# Then we move to this state, the current position being changed + the states is alternated.
def construct_class_a(N: int, P_0: float, distance_magnitude: float,
                      allow_red_start: bool, starting_point: tuple,
                      allow_repeat: bool,
                      max_radius: float,
                      change_polarity: bool) -> list[tuple[float, float, float, float]]:

    #Initialise the graph
    graph_coordinates = {}
    graph_coordinates[0] = starting_point

    #Get the new point
    new_point = starting_point

    #Assign default values
    red_dot_currently = False
    left_layout = False

    #Initialise the state
    random_start_point_type = random.uniform(0, 1)
    if allow_red_start:
        if random_start_point_type < 1/3:
            red_dot_currently = True
        elif random_start_point_type < 2/3:
            left_layout = True
    else:
        if random_start_point_type < 1/2:
            left_layout = True

    #Does the time dimension exist?
    incorporate_time = False
    incorporate_polarity = False
    if len(starting_point) > 2:
        incorporate_time = True
        if len(starting_point) == 4:
            incorporate_polarity = True


    while len(graph_coordinates) < N:
        new_point, graph_coordinates, left_layout, red_dot_currently = class_a_step(new_point, graph_coordinates,
                                                                                    distance_magnitude, P_0,
                                                                                    left_layout, red_dot_currently,
                                                                                    incorporate_time, incorporate_polarity,
                                                                                    allow_repeat, max_radius,
                                                                                    change_polarity)
    return graph_coordinates

#Constructing the kagome graph
# Explanation: The coordinates (x,y) are kept and each point has an associated state with it. This state assings the
# weight of the direction we want to go after the comptutation. Since this is a triangular lattice, we have
# 6 possible directions, which in 2D it is the 6 roots of unity on a complex plane.
# For each coordinate we have 2 different options: either it is a red dot (i.e. outside of the typical bounds)
# or it is a blue dot (i.e. inside the expected region)
# If we are in the red dot, that is, in the center of a hexagon, then we have equal probability of jumping along each node
# If we are in the blue dot, then we have a state to keep in mind. This state can be 3 possible types.
#This is based on the following:
    #From 0 pick 1 -> 2
    #From 0 pick 2 -> 1
    #From 1 pick 0 -> 2
    #From 1 pick 2 -> 0
    #From 2 pick 1 -> 0
    #From 2 pick 0 -> 1
    #This can be summerised as:
    #f(0, 1) = 2
    #f(0, 2) = 1
    #f(1, 2) = 0
    #This condition is fulfilled with the simple function: (2*(layout_type + next_node) % 3)
# Then we move to this state, the current position being changed + the states is alternated.
def construct_class_b(N: int, P_0: float, distance_magnitude: float,
                      allow_red_start: bool, starting_point: tuple,
                      allow_repeat: bool,
                      max_radius: float,
                      change_polarity: bool) -> list[tuple[float, float, float, float]]:

    #Initialise the graph
    graph_coordinates = {}
    graph_coordinates[0] = starting_point

    #Get the new point
    new_point = starting_point

    #Assign default values
    red_dot_currently = False
    layout_type = 2

    #Initialise the state
    random_start_point_type = random.uniform(0, 1)

    if allow_red_start:
        if random_start_point_type < 1/4:
            red_dot_currently = True
        elif random_start_point_type < 2/4:
            layout_type = 0
        elif random_start_point_type < 3/4:
            layout_type = 1
    else:
        if random_start_point_type < 1/3:
            layout_type = 0
        elif random_start_point_type < 2/3:
            layout_type = 1


    #Does the time dimension exist?
    incorporate_time = False
    incorporate_polarity = False
    if len(starting_point) > 2:
        incorporate_time = True
        if len(starting_point) == 4:
            incorporate_polarity = True

    while len(graph_coordinates) < N:
        new_point, graph_coordinates, layout_type, red_dot_currently = class_b_step(new_point, graph_coordinates,
                                                                                    distance_magnitude, P_0,
                                                                                    layout_type, red_dot_currently,
                                                                                    incorporate_time, incorporate_polarity,
                                                                                    allow_repeat, max_radius, change_polarity)
    return graph_coordinates



##################################################
###################### Main ######################
##################################################

#### Dataset creation of the two elements
def create_dataset(num_nodes: int = 10, num_per_set: int = 100, p_0: float = 0.0, dist: float = 5.0,
                   allow_red_start: bool = True, starting_point: tuple = (0,0,0,1),
                   seed: int = 0, allow_repeat: bool = True,
                   max_radius: float = 0, change_polarity: bool = False) -> list[tuple[dict[int, tuple[float, float, float, float]], int]]:


    #Use the randomness
    if seed != 0:
        random.seed(seed)


    #Now create the dataset
    graphs_dataset = []

    for _ in range(num_per_set):

        new_graph_a = construct_class_a(num_nodes, p_0, dist, allow_red_start, starting_point, allow_repeat, max_radius, change_polarity)
        new_graph_b = construct_class_b(num_nodes, p_0, dist, allow_red_start, starting_point, allow_repeat, max_radius, change_polarity)

        graphs_dataset.append((new_graph_a, 1))
        graphs_dataset.append((new_graph_b, -1))

    random.shuffle(graphs_dataset)
    return graphs_dataset




###################################################################################################################################################################
############################################################################ WL Graphs ############################################################################
###################################################################################################################################################################

#This should return 2 different graphs, that should yield the same result on the Weisfeiler Lehman action
def create_same_wl_graphs(distance: float, polygon_degree: int = 3):

    #Create first graph
    polygon_angle_1 = (2*np.pi)/(polygon_degree)
    circle_radius_1 = distance/(2*np.sin(polygon_angle_1/2)) #Use the sinusoidal theorem to find the radius

    left_center_x_1 = -((distance/2) + circle_radius_1)
    right_center_x_1 = ((distance/2) + circle_radius_1)

    #Create the coordinates now
    graph_1_coordinates = []


    for i in range(polygon_degree):
        left_coordinates = [left_center_x_1 + circle_radius_1*np.cos((i+1)*polygon_angle_1), circle_radius_1*np.sin((i+1)*polygon_angle_1)]
        graph_1_coordinates.append(left_coordinates)

    for i in range(polygon_degree):
        right_coordinates = [right_center_x_1 + circle_radius_1*np.cos(np.pi + i*polygon_angle_1), circle_radius_1*np.sin(np.pi + i*polygon_angle_1)]
        graph_1_coordinates.append(right_coordinates)


    polygon_degree_2 = int(polygon_degree + 1)
    polygon_angle_2 = (2*np.pi)/(polygon_degree_2)
    starting_angle = polygon_angle_2/2
    circle_radius_2 = distance/(2*np.sin(starting_angle))

    left_center_x_2 = -circle_radius_2*np.cos(starting_angle)
    right_center_x_2 = circle_radius_2*np.cos(starting_angle)


    #Create the coordinates now
    graph_2_coordinates = []

    for i in range(polygon_degree_2):
        left_coordinates = [left_center_x_2 + circle_radius_2*np.cos(starting_angle + (i+1)*polygon_angle_2), circle_radius_2*np.sin(starting_angle +(i+1)*polygon_angle_2)]
        graph_2_coordinates.append(left_coordinates)

    for i in range(polygon_degree_2 - 2):
        right_coordinates = [right_center_x_2 + circle_radius_2*np.cos(np.pi + starting_angle + (i+1)*polygon_angle_2), circle_radius_2*np.sin(np.pi + starting_angle + (i+1)*polygon_angle_2) ]
        graph_2_coordinates.append(right_coordinates)


    return graph_1_coordinates, graph_2_coordinates

