import random
import numpy as np
import collections

##################################
############# Tools ##############
##################################

def create_random_time_list(parameter_lists_to_generate: int, max_t_allowed: int) -> list[list[int]]:

    random_time_lists = []

    lower_bound = 16
    n = int(max_t_allowed/4)
    for _ in range(parameter_lists_to_generate):

        #Create the cumulative distribution
        t1 = random.randint(lower_bound, n)
        t2 = random.randint(lower_bound, n)
        t3 = random.randint(lower_bound, n)
        t4 = random.randint(lower_bound, n)
        random_time_lists.append([t1, t2, t3, t4])

    return random_time_lists


## Create the adjacency matrix of the dataset: A
def create_adjacency_matrix(events: dict[str,tuple[float,float]], threshold_radius: float = 5.1) -> list[list[int]]:


    keys = list(events.keys())
    n = len(keys)

    #Create a zero adjacency matrix
    adjacency_matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(i):

            #Add an entry only if it is within radius
            if np.sqrt((events[keys[i]][0] - events[keys[j]][0]) ** 2 +
                       (events[keys[i]][1] - events[keys[j]][1]) ** 2) < threshold_radius:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1


    return adjacency_matrix


## Create the adjacency matrix of the dataset: A
def create_adjacency_matrix_3D(events: dict[str, tuple[float,float,float]], threshold_radius: float = 5.1,
                               time_weight: float = 1.0) -> np.ndarray:


    keys = list(events.keys())
    n = len(keys)

    #Create a zero adjacency matrix
    adjacency_matrix = [[0]*n for _ in range(n)]

    for i in range(n):
        for j in range(i):

            #Do the graph checks
            if (len(events[keys[i]]) < 3):
                raise Exception("All the graphs should have a third coordinate")

            #Add an entry only if it is within radius
            if np.sqrt((events[keys[i]][0] - events[keys[j]][0]) ** 2 +
                       (events[keys[i]][1] - events[keys[j]][1]) ** 2 +
                       (time_weight*(events[keys[i]][2] - events[keys[j]][2])) ** 2) < threshold_radius:

                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1


    return np.array(adjacency_matrix)


## Create the adjacency list, where same indecies contain an edge
def create_adjacency_list_3D(events: dict[str, tuple[float,float,float]], threshold_radius: float = 5.1,
                               time_weight: float = 1.0, include_self_term: bool = False) -> np.ndarray:


    keys = list(events.keys())
    n = len(keys)

    #Create a zero adjacency matrix
    adjacency_list_source = []
    adjacency_list_target = []

    for i in range(n):
        for j in range(n):

            #Do the graph checks
            if (len(events[keys[i]]) < 3):
                raise Exception("All the graphs should have a third coordinate")

            #Add an entry only if it is within radius
            if np.sqrt((events[keys[i]][0] - events[keys[j]][0]) ** 2 +
                       (events[keys[i]][1] - events[keys[j]][1]) ** 2 +
                       (time_weight*(events[keys[i]][2] - events[keys[j]][2])) ** 2) < threshold_radius and (include_self_term or i != j):

                adjacency_list_source.append(keys[i])
                adjacency_list_target.append(keys[j])



    return np.array([adjacency_list_source, adjacency_list_target])


#Find lower and upper clipping bound, given distribution
def find_clipping_bounds(state_vector_l: np.ndarray[np.ndarray], clipped_percentile: int = 20, rounding_num: bool = False):

    lower_clipping_percentile = clipped_percentile/2
    upper_clipping_percentile = 100 - lower_clipping_percentile

    lower_bound = upper_bound = 0
    if rounding_num:
        lower_bound = np.round(np.percentile(state_vector_l, lower_clipping_percentile))
        upper_bound = np.round(np.percentile(state_vector_l, upper_clipping_percentile))
    else:
        lower_bound = np.percentile(state_vector_l, lower_clipping_percentile)
        upper_bound = np.percentile(state_vector_l, upper_clipping_percentile)

    return lower_bound, upper_bound



#Count the votes in an array
def count_vote(vote_array: np.ndarray) -> np.ndarray:

    unique_entries, counts = np.unique(vote_array, return_counts=True)

    return unique_entries[np.argmax(counts)]

#Reduces the vector to the minimum value
def reduce_min_func(state_vector):

    val_to_reduce = np.min(state_vector) - 1
    return state_vector - val_to_reduce


#A function to apply the clipping strategy
def apply_clipping_func(state_vector_l: np.ndarray):

    lower_clip, upper_clip = find_clipping_bounds(state_vector_l)

    #Create the masks
    lower_mask = state_vector_l < lower_clip
    upper_mask = state_vector_l > upper_clip

    #Apply the clipping
    state_vector_l[lower_mask] = lower_clip
    state_vector_l[upper_mask] = upper_clip

    #Change the upper and the lower bounds
    lower_bound = lower_clip
    upper_bound = upper_clip


    return state_vector_l, lower_bound, upper_bound

#A function that applies the quantisation binning strategy
def apply_quantisation_func(state_vector_l: np.ndarray[float], lower_bound: float, upper_bound: float, B: int):


    #Create the bins
    bins = np.linspace(lower_bound, upper_bound, B+1)
    for i in range(len(bins) - 1):

        #Create the bin mask
        lower_mask = state_vector_l >= bins[i]
        upper_mask = state_vector_l <= bins[i+1]
        bin_mask = lower_mask & upper_mask

        #Alter the values
        state_vector_l[bin_mask] = bins[i]


    return state_vector_l


#Convert the list to dictionary
def convert_to_dict(events_list: list[tuple[int,int,float,int]]) -> dict[int,tuple[int,int,float,int]]:

    events_dict = {i: event for i, event in enumerate(events_list)}

    return events_dict



def count_events_in_window(events: dict[int, tuple[float,float,float,float]] | list[tuple[float,float,float,float]],
                           t_w: float, delta_t: float, allow_repeat: bool) -> int:

        start_window = t_w
        end_window = t_w + delta_t

        num_events = 0
        if isinstance(events, list):

            for graph_coords in events:
                if ((start_window <= graph_coords[2]) and (graph_coords[2] < end_window)):
                    num_events += 1
        elif isinstance(events, dict):

            for graph_coords in events.values():
                if ((start_window <= graph_coords[2]) and (graph_coords[2] < end_window)):
                    num_events += 1

        else:
            raise Exception(f"Events must be either a dicitonary or a list. Instead, events is of type {type(events)}")

        return num_events

def find_max_number_of_events(graphs_dataset: dict[int, tuple[float,float,float,float]] | list[tuple[float,float,float,float]],
                              time: float, delta_t: float, allow_repeat: bool = True):

        max_num_events = 0
        for graph_coords, _ in graphs_dataset:
            current_num_events = count_events_in_window(graph_coords, time, delta_t, allow_repeat)
            if max_num_events < current_num_events:
                max_num_events = current_num_events

        return max_num_events

#######################################################
### Subsampling on pixels with the same coordinates ###
#######################################################
def subsample_spike_per_pixel(graph_in_time_window: dict[int, tuple[int,int,float,int]]) -> tuple[dict[int,tuple[int,int,float,int]],dict[tuple[float,float],list[int]]]:


    # This collects the previous coordinates. This list should collect only the unique events
    # Key = event number
    # value = (x, y, t) for the coordinate
    coordinate_collect = {}

    # A dictionary that collects the previous coordinates
    # key = 2D tuple of the x and y coordinates
    # value = list of events with coinciding x and y coordinate
    passed_coordinates = {}

    # This collects the polarity,
    # key = 2D tuple of the x and y coordinates
    # value = an integer, storing the polarity to be established.
    collect_polarity = {}


    for event_id, event_coordinate in graph_in_time_window.items():
        x_coord = event_coordinate[0]
        y_coord = event_coordinate[1]
        coords_2D = (x_coord, y_coord)

        polarity_weight_to_add = 1 if event_coordinate[3] == 1 else -1
        if (coords_2D in passed_coordinates):

            #Add the new event id to the previous list
            prev_events_at_current_coordinate = passed_coordinates[coords_2D]
            prev_events_at_current_coordinate.append(event_id)
            passed_coordinates[coords_2D] = prev_events_at_current_coordinate

            #Increment the polarity

            collect_polarity[coords_2D] += polarity_weight_to_add # The value of the polarity
        else:

            #First add the coordinate to the previous coordinates
            event_at_coordinate_list = [event_id]
            passed_coordinates[coords_2D] = event_at_coordinate_list

            #Add the new coordinate
            coordinate_collect[event_id] = (x_coord, y_coord, event_coordinate[2])

            #Collect the polarity, adding a small weight to the initial input
            collect_polarity[coords_2D] = 1.01 * polarity_weight_to_add


    new_graph_in_time_window = {} #The rewritten graph, overwriting the coordinates


    #Now create the new graph
    for event_id, event_coordinates in coordinate_collect.items():

        #Get only the 3D coordinates
        x_coord, y_coord, init_t_coord = event_coordinates

        #Find the polarity
        polarity = 1 if collect_polarity[(x_coord, y_coord)] > 0 else 0 # We need to do only 1 or 0 polarity

        #Write the new element
        new_graph_in_time_window[event_id] = (x_coord, y_coord, init_t_coord, polarity)


    return new_graph_in_time_window, passed_coordinates


def expand_spike_per_pixel(graph: dict[int, tuple[int,int,float,int]],
                           node_features: list[dict[int, float]],
                           coordinates_events: dict[tuple[float, float], list[int]],
                           convert_key_to_string: bool = True) -> list[dict[int, float]]:



    returned_node_features = []
    for layer_features in node_features:

        returned_node_features_layer = {}

        for event_coords in graph.values():
            x_coord, y_coord, _, _ = event_coords

            coordinate_key = (x_coord, y_coord)

            event_keys = coordinates_events[coordinate_key]

            existing_event_key = event_keys[0]
            for event_key in event_keys:
                if convert_key_to_string and (isinstance(existing_event_key, int) or isinstance(existing_event_key, float)):
                    existing_event_key = str(existing_event_key)
                returned_node_features_layer[event_key] = layer_features[existing_event_key]

        returned_node_features_layer = dict(collections.OrderedDict(sorted(returned_node_features_layer.items())))
        returned_node_features.append(returned_node_features_layer)


    return returned_node_features

def remove_near_time_events(events: list[tuple[float,float,float,float]], near_time_range: float) -> list[tuple[float,float,float,float]]:

    new_events = []
    last_occur_dict = {}
    for event in events:
        event_time = event[2]

        #Store the coordinates
        event_x = event[0]
        event_y = event[1]

        coords = (event_x, event_y)
        if coords in last_occur_dict:

            if event_time - last_occur_dict[coords] > near_time_range:
                new_events.append(event)

            #Update the last time
            last_occur_dict[coords] = event_time

        else:
            last_occur_dict[coords] = event_time
            new_events.append(event)

    return new_events


def sigmoid(x: float, input_factor: float) -> float:

    return 1/(1 + np.exp(-input_factor * x))