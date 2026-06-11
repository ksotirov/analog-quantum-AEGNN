import numpy as np
import pulser
from dataclasses import replace
from pulser.channels.dmm import DMM

#User defined libraries
from . import qaaegnn
from . import tools

### Lessons from classical event-based kernel.
### 1) First features are extracted
### How is this done in more detail
###### a) Firstly we extract the coordinates of every graph
###### b) For every graph, we slide a window. In each window we do
######### i)   Capture the nodes inside the graph
######### ii)  Creates the adjacency matrix/list(depending on the process)
######### iii) Initialises the features
######### iv)  Extracts the feature
######### v)   Preserve the necessary features.
### 2) The kernel is performed, to classify the output, based on the extracted features

class QuantumGraph():

    def __init__(self, time: list[float] | float, delta_t: float,
                  quantum_evolution_time: list[int],
                  recenter: bool, device_type: str,
                  legacy_features_init_type: str,
                  one_graph_window: bool = False, dimensions: int = 2,
                  normalise_xy: bool = False,
                  max_num_events: int = 10,
                  subsample_spike_per_pixel: bool = False,
                  linear_multiplicative_time_c: float = 0,
                  split_time_in_middle: bool = False,
                  number_of_quantum_shots: int = 100,
                  remove_close_points: bool = False,
                  allow_noise: bool = False, noise_model_params = None,
                  detuning_encoding: bool = False):


        self.time = time
        self.delta_t = delta_t
        self.q_evol_time = quantum_evolution_time
        self.recenter = recenter
        self.device_type = device_type
        self.legacy_features_init_type = legacy_features_init_type

        self.one_graph_window = one_graph_window

        if self.one_graph_window:
            self.readout_graph_features = self.readout_graph_features_one_window
        else:
            self.readout_graph_features = self.readout_graph_features_all_windows

        if dimensions < 2 or dimensions > 3:
            raise Exception("Only 2D and 3D graphs are allowed")

        self.dimensions = dimensions

        self.max_num_events = max_num_events

        self.normalise_xy = normalise_xy

        self.subsample_spike_per_pixel = subsample_spike_per_pixel


        #This is in case we have 2 dimensions, but time would be included
        self.linear_multiplicative_time_c = linear_multiplicative_time_c

        #In this case, we care only about the time relative to the time frame
        self.normalise_time_frame = False
        if dimensions == 2 and (linear_multiplicative_time_c != 0):
            self.normalise_time_frame = True

        self.split_time_in_middle = split_time_in_middle

        self.number_of_quantum_shots = number_of_quantum_shots
        self.remove_close_points = remove_close_points

        self.allow_noise = allow_noise
        self.noise_model_params = noise_model_params

        self.detuning_encoding = detuning_encoding

    # b) For every graph, we slide a window. In each window we do
    #The function executes the neuromorphic step of a single graph.
    def readout_graph_features_all_windows(self, events: dict[int, tuple[float, float, float, float]]):


        #Initialise node features
        node_features = []

        final_layer_dict = {}


        #Iterate through all the possible windows
        for t in self.time:

            ###### Create graph ######

            # i)   Capture the nodes inside the graph
            #Initialise the nodes, and hence the graph
            subgraph_in_time_window = self.slide_window(events, t, alter_polarity=True) #Creates the vertices. NOTE: We need to change the polarity of -1 to 0


            #### Create graph (End)###

            #Initialise the features for this subgraph
            subgraph_features = {}
            match self.legacy_features_init_type:
                case 'default':
                    subgraph_features = self.initialise_features_initial(subgraph_in_time_window)
                case 'last_layer':
                    subgraph_features = self.initialise_features_last_layer(subgraph_in_time_window, final_layer_dict)
                case 'all_one':
                    subgraph_features = self.initialise_features_all_one(subgraph_in_time_window)
                case '_':
                    raise Exception("Unknown feature initialisation type")

            #Compute the graph features for this step
            subgraph_node_features, subgraph_final_features = self.get_node_features_quantum_layer(subgraph_in_time_window, subgraph_features)

            print(f"Executed run for time step {t}")

            ##Add the new features to the graph/update the existing features
            node_features.append(subgraph_node_features)

            final_layer_dict |= subgraph_final_features

        return node_features

    def readout_graph_features_one_window(self, events: dict[int, tuple[float, float, float, float]] | list[tuple[float, float, float, float]], final_layer_dict: dict[int,np.float64]):


        ###### Create graph ######

        # i)   Capture the nodes inside the graph

        if isinstance(events, list):
            events = tools.convert_to_dict(events)

        #Initialise the nodes, and hence the graph
        subgraph_in_time_window = self.slide_window(events, self.time, alter_polarity=True) #Creates the vertices. NOTE: We need to change the polarity of -1 to 0


        coordinates_events = None
        if self.subsample_spike_per_pixel:
            subgraph_in_time_window, coordinates_events = tools.subsample_spike_per_pixel(subgraph_in_time_window)
        #### Create graph (End)###


        #Initialise the features for this subgraph
        subgraph_features = {}
        match self.legacy_features_init_type:
            case 'default':
                subgraph_features = self.initialise_features_initial(subgraph_in_time_window)
            case 'last_layer':
                subgraph_features = self.initialise_features_last_layer(subgraph_in_time_window, final_layer_dict)
            case 'all_one':
                subgraph_features = self.initialise_features_all_one(subgraph_in_time_window)
            case '_':
                    raise Exception("Unknown feature initialisation type")

        #Compute the graph features for this step
        subgraph_features_histogram, subgraph_features_list = self.get_node_features_quantum_layer(subgraph_in_time_window, subgraph_features)

        if self.subsample_spike_per_pixel:
            subgraph_features_list = tools.expand_spike_per_pixel(subgraph_in_time_window, subgraph_features_list, coordinates_events)
        print(f"Executed run for time step {self.time}")


        return subgraph_features_histogram, subgraph_features_list

    ##################################
    ##### Neuromorphic functions #####
    ##################################

    #This method returns the coordinates used in this time window, that are used to construct the graph
    #This is based on the time coordinate (i.e. the last coordinate)
    def slide_window(self, events: dict[int, tuple[float,float,float,float]],
                     t_w: float,
                     alter_polarity: bool = False) -> dict[int,tuple[float,float,float,float]]:

        start_window = t_w
        end_window = t_w + self.delta_t

        event_subset = {}
        for event_key, graph_coords in events.items():
            if ((start_window <= graph_coords[2]) and (graph_coords[2] < end_window)):

                if alter_polarity and graph_coords[3] == -1:
                    x_coord, y_coord, t_coord, _ = graph_coords
                    event_subset[event_key] = (x_coord, y_coord, t_coord, 0) #Change the last coordinate to 0
                else:
                    event_subset[event_key] = graph_coords


        return event_subset


    ##################################
    ###### Node Initialisation #######
    ##################################


    #Features are initialised as polarity
    def initialise_features_initial(self, subgraph: dict[int,tuple[float,float,float,int]]) -> dict[str,int]:


        #Initialise graph features
        subgraph_features = {}


        #Extract the key for the dictionary
        dict_keys_per_run = subgraph.keys()

        for key in dict_keys_per_run:
            subgraph_features[str(key)] = subgraph[key][3] # Use the polarity as the key

        return subgraph_features

    #Features are initialised as last layer
    def initialise_features_last_layer(self, subgraph: dict[int,tuple[float,float,float,int]], final_layer_dict:dict[int,np.float64]) -> dict[str,int]:

        #Initialise graph features
        subgraph_features = {}


        #Extract the key for the dictionary
        dict_keys_per_run = subgraph.keys()

        for key in dict_keys_per_run:
            if key in final_layer_dict:
                subgraph_features[str(key)] = final_layer_dict[key] #Reuse the features from the previous run
            else:
                subgraph_features[str(key)] = subgraph[key][3] # Use the polarity as the key


        return subgraph_features

    #All features are initialised to 1
    def initialise_features_all_one(self, subgraph: dict[int,tuple[float,float,float,int]]):

        #Initialise graph features
        subgraph_features = {}


        #Extract the key for the dictionary
        dict_keys_per_run = subgraph.keys()

        for key in dict_keys_per_run:
            subgraph_features[str(key)] = 1 #Initialisa this all to 1

        return subgraph_features

    #A function to process the quantum information!
    def get_node_features_quantum_layer(self, graph_to_process: dict[int, tuple[float, float, float, float]],
                                        node_features: dict[str,int]):

        p_dists = return_features = None

        #Get the size of the graph
        n = len(graph_to_process)

        if n > 0:

            #Create the device
            device = pulser.AnalogDevice #NOTE: This can change. Find a way to fix it.

            rabi_freq = 2*np.pi
            detuning_freq = 0.7*rabi_freq

            if self.device_type == 'mock':
                device = pulser.MockDevice
                rabi_freq = 12.5*np.pi
                detuning_freq = 0.7*rabi_freq

            #If we use the detuning map, we need to alter the device, to allow for Detunable Maps
            if self.device_type != 'mock' and self.detuning_encoding:

                #Define the Detuning Map
                dmm = DMM(
                    clock_period=4,
                    min_duration=16,
                    max_duration=2**26,
                    mod_bandwidth=8,
                    bottom_detuning=-2 * np.pi * 20,  # detuning between 0 and -20 MHz
                    total_bottom_detuning=-2 * np.pi * 2000,  # total detuning
                )

                #Modify the device
                device = replace(
                   device.to_virtual(),
                   dmm_objects =(dmm,DMM()),
                   reusable_channels=True,
                )

            #Convert the 4D tuple into a 2D one
            #Obtain only the xy coordinate
            coord_graph = {}

            ######### ii)  Creates the adjacency matrix/list(depending on the process)

            #Multiply neighboring pixels by the rydberg coefficient
            mult_coeff = 1
            if self.normalise_xy:
                mult_coeff = device.rydberg_blockade_radius(rabi_freq)/np.sqrt(2)



            for event_id, event_coordinate in graph_to_process.items():
                x_coord, y_coord, t_coord, _ = event_coordinate

                if self.normalise_time_frame:

                    if self.split_time_in_middle:
                        t_coord -= (self.time + self.delta_t/2)
                    else:
                        t_coord -= self.time

                coord_graph[event_id] = (x_coord, y_coord, t_coord)


            #Extract the features!!!
            #The features are essentailly a NxM vector, where N (rows) is the different times of the execution and M = Features!
            p_dists, node_features_all_time = qaaegnn.sample_single_distribution_v3(coord_graph, self.max_num_events, device,
                                                                                initial_features=node_features,t_list=self.q_evol_time,
                                                                                recenter=self.recenter, rabi_freq=rabi_freq,detuning_freq=detuning_freq,
                                                                                dimensions=self.dimensions,
                                                                                num_of_samples=self.number_of_quantum_shots,
                                                                                mult_coeff=mult_coeff, linear_multiplicative_time_c=self.linear_multiplicative_time_c,
                                                                                remove_close_points=self.remove_close_points,
                                                                                allow_noise=self.allow_noise, noise_model_params=self.noise_model_params,
                                                                                detuning_encoding=self.detuning_encoding)

            #For now, lets obtain the final evolution one. It can be changed in the future
            return_features = None
            if self.one_graph_window:
                return_features = node_features_all_time #Return all the nodes, in case only one graph window is processed at a time
            else:
                return_features = node_features_all_time[-1] #Return only the window after the entire computation, if we process all the time windows

        else:
            p_dists = []
            for _ in range(len(self.q_evol_time)):
                p_dist_per_time = np.zeros(self.max_num_events + 1)
                p_dist_per_time[0] = 1 #Fix error!
                p_dists.append(p_dist_per_time)

            return_features = [{} for _ in range(len(self.q_evol_time))]

        return p_dists, return_features