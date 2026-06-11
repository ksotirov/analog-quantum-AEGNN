####### External libraries #######
import numpy as np

####### Defined libraries ########
from . import kernel as krn
from . import tools
from . import normalisation_enum as ne


##################################
##### Neuromorphic functions #####
##################################

#This method returns the coordinates used in this time window, that are used to construct the graph
#This is based on the time coordinate (i.e. the last coordinate)
def slide_window(events: dict[int, tuple[float,float,float,float]],
                 t_w: float, delta_t: float) -> dict[int,tuple[float,float,float,float]]:

    start_window = t_w
    end_window = t_w + delta_t

    event_subset = {}
    for event_key, graph_coords in events.items():
        if ((start_window <= graph_coords[2]) and (graph_coords[2] < end_window)):
            event_subset[event_key] = graph_coords

    return event_subset


##################################
########## Layer Types ###########
##################################

#Version 0) Simply multiply the matrix
def layer_ver_0(subgraph_adj_matrix_numpy: np.ndarray[np.ndarray], state_vector_l_prev: list[float]):


    #NOTE: This is not the optimal method, but it separates things into segmented functions
    n = subgraph_adj_matrix_numpy.shape[0]

    #Add the ones element to the adjacency matrix, to include the self term
    I = np.eye(n)
    graph_adjacency_matrix = np.add(subgraph_adj_matrix_numpy, I)

    #Initially, the current state vector is created, by mutliplying the previous state vector with the adjacency matrix
    state_vector_l = np.matmul(graph_adjacency_matrix, state_vector_l_prev)

    return state_vector_l


#Version 1) Find the minimum and subtract
def layer_ver_1(subgraph_adj_matrix_numpy: np.ndarray[np.ndarray], state_vector_l_prev: list[float]) -> np.ndarray:

    #NOTE: This is not the optimal method, but it separates things into segmented functions
    n = subgraph_adj_matrix_numpy.shape[0]

    #Add the ones element to the adjacency matrix, to include the self term
    I = np.eye(n)
    graph_adjacency_matrix = np.add(subgraph_adj_matrix_numpy, I)

    #Initially, the current state vector is created, by mutliplying the previous state vector with the adjacency matrix
    state_vector_l = np.matmul(graph_adjacency_matrix, state_vector_l_prev)

    state_vector_l = tools.reduce_min_func(state_vector_l)

    return state_vector_l

#Version 1.1) Find the minimum and subtract + add small extra for the current node
def layer_ver_1_1(subgraph_adj_matrix_numpy: np.ndarray[np.ndarray], state_vector_l_prev: list[float]) -> np.ndarray:

    #NOTE: This is not the optimal method, but it separates things into segmented functions
    n = subgraph_adj_matrix_numpy.shape[0]

    #Add the ones and an extra element to the adjacency matrix, to include the self term
    eI = 1.01*np.eye(n)
    graph_adjacency_matrix = np.add(subgraph_adj_matrix_numpy, eI)

    #The current state vector is created, by mutliplying the previous state vector with the adjacency matrix
    state_vector_l = np.matmul(graph_adjacency_matrix, state_vector_l_prev)

    state_vector_l = tools.reduce_min_func(state_vector_l)

    return state_vector_l

#Version 2) Check only number based on order of elements
def layer_ver_2(subgraph_adj_matrix_numpy: np.ndarray[np.ndarray], state_vector_l_prev: list[float]) -> np.ndarray[np.ndarray]:

    #NOTE: This is not the optimal method, but it separates things into segmented functions
    n = subgraph_adj_matrix_numpy.shape[0]

    #Add the ones element to the adjacency matrix, to include the self term
    I = np.eye(n)
    graph_adjacency_matrix = np.add(subgraph_adj_matrix_numpy, I)


    #Initially, the current state vector is created, by mutliplying the previous state vector with the adjacency matrix
    state_vector_l = np.matmul(graph_adjacency_matrix, state_vector_l_prev)


    #Then it is ordered, and only the necessary elements are taken into account
    ordered_state_vector_state = list(set(state_vector_l))

    ordered_state_vector_state.sort()

    for i, element in enumerate(ordered_state_vector_state):
        state_vector_l[np.where(state_vector_l == element)] = i + 1

    return state_vector_l

#Version 3) Aggregate as usual + clipping and uniform quantisation
def layer_ver_3(subgraph_adj_matrix_numpy: np.ndarray[np.ndarray], state_vector_l_prev: list[float],
                B: int, apply_clipping: bool, reduce_min: bool) -> np.ndarray[np.float64]:

    ##### Aggregate (Begin) #####

    #NOTE: This is not the optimal method, but it separates things into segmented functions
    n = subgraph_adj_matrix_numpy.shape[0]

    #Add the ones element to the adjacency matrix, to include the self term
    I = np.eye(n)
    graph_adjacency_matrix = np.add(subgraph_adj_matrix_numpy, I)

    #Initially, the current state vector is created, by mutliplying the previous state vector with the adjacency matrix
    state_vector_l = np.matmul(graph_adjacency_matrix, state_vector_l_prev)

    ###### Aggregate (End) ######

    if reduce_min:
        state_vector_l = tools.reduce_min_func(state_vector_l)


    ###### Clipping (Begin) #####

    lower_bound = state_vector_l.min()
    upper_bound = state_vector_l.max()

    if apply_clipping:
        state_vector_l, lower_bound, upper_bound = tools.apply_clipping_func(state_vector_l)

    ####### Clipping (End) ######


    #### Quantisation (Begin) ###
    state_vector_l = tools.apply_quantisation_func(state_vector_l, lower_bound, upper_bound, B)

    ##### Quantisation (End) ####

    return state_vector_l


def layer_ver_aegnn(subgraph_adj_source_target_numpy: np.ndarray,
                    node_features: dict[int,int],
                    past_nodes: set) -> tuple[dict[int,int],set]:

    '''
    * subgraph_adj_source_target_numpy -> The 2D np.ndarray. It contains a 2 lists.
      One is the sources i.e. the point where the edges start from.
      The other one is the target i.e. the point where the edge ends.

    * tracking_features -> used to see what features are used in the current layer.
      Initially, it is only the new nodes

    * node_features_current_layer -> What are all the features in the current layer. This tracks them for the entire computation
    '''

    ##### Aggregate (Begin) #####

    #First get the two nodes you need
    source_nodes = subgraph_adj_source_target_numpy[0]
    target_nodes = subgraph_adj_source_target_numpy[1]

    current_node_features = {}

    #Copy the values
    for event, polarity in node_features.items():
        current_node_features[event] = polarity

    current_nodes = set({})

    for node in past_nodes:

        #Get the list indecies of the past nodes
        node_indecies = np.where(source_nodes == node)[0]

        for index in node_indecies:

            source_graph_node = source_nodes[index]
            target_graph_node = target_nodes[index]
            current_node_features[source_graph_node] += node_features[target_graph_node]

            #Save the newly imported node
            current_nodes.add(target_graph_node)


    #Add the new nodes for the next iteration
    past_nodes.update(current_nodes)

    ###### Aggregate (End) ######


    return current_node_features, past_nodes

#AEGNN Apply non-linearity
def layer_ver_aegnn_v2(subgraph_adj_source_target_numpy: np.ndarray,
                       node_features: dict[int,int],
                       past_nodes: set, alpha: float) -> tuple[dict[int,int],set]:

    '''
    * subgraph_adj_source_target_numpy -> The 2D np.ndarray. It contains a 2 lists.
      One is the sources i.e. the point where the edges start from.
      The other one is the target i.e. the point where the edge ends.

    * tracking_features -> used to see what features are used in the current layer.
      Initially, it is only the new nodes

    * node_features_current_layer -> What are all the features in the current layer. This tracks them for the entire computation
    '''

    ##### Aggregate (Begin) #####

    #First get the two nodes you need
    source_nodes = subgraph_adj_source_target_numpy[0]
    target_nodes = subgraph_adj_source_target_numpy[1]

    current_node_features = {}

    #Copy the values
    for event, polarity in node_features.items():
        current_node_features[event] = polarity

    current_nodes = set({})

    for node in past_nodes:

        #Get the list indecies of the past nodes
        node_indecies = np.where(source_nodes == node)[0]

        for index in node_indecies:

            source_graph_node = source_nodes[index]
            target_graph_node = target_nodes[index]

            if target_graph_node != source_graph_node:
                current_node_features[source_graph_node] += node_features[target_graph_node]

            #Save the newly imported node
            current_nodes.add(target_graph_node)

    #Apply the non-linearity only to the past nodes
    for node in past_nodes:
        current_node_features[node] = (2*tools.sigmoid(current_node_features[node], alpha) - 1) #Introduce it in the range between 1 and -1


    #Add the new nodes for the next iteration
    past_nodes.update(current_nodes)

    ###### Aggregate (End) ######



    return current_node_features, past_nodes



##################################
###### Node Initialisation #######
##################################


#Features are initialised as polarity
def initialise_features_ver_1(subgraph: dict[int,tuple[float,float,float,int]], final_layer_dict:dict[int,np.float64]) -> dict:


    #Initialise graph features
    subgraph_features = {}


    #Extract the key for the dictionary
    dict_keys_per_run = subgraph.keys()

    for key in dict_keys_per_run:
        value = subgraph[key][3]

        #Change the features to be 1/-1 instead of 1/0
        if value == 0:
            value = -1

        subgraph_features[key] = value # Use the polarity as the key

    return subgraph_features

#Features are initialised as last layer
def initialise_features_ver_2(subgraph: dict[int,tuple[float,float,float,int]], final_layer_dict:dict[int,np.float64]) -> dict:

    #Initialise graph features
    subgraph_features = {}


    #Extract the key for the dictionary
    dict_keys_per_run = subgraph.keys()

    for key in dict_keys_per_run:
        if key in final_layer_dict:
            subgraph_features[key] = final_layer_dict[key] #Reuse the features from the previous run
        else:
            subgraph_features[key] = subgraph[key][3] # Use the polarity as the key


    return subgraph_features


#Features are initialised as last layer
def initialise_features_ver_3(subgraph: dict[int,tuple[float,float,float,int]], final_layer_dict:dict[int,np.float64],
                              lower_val: float = -1.0, upper_val: float = 1.0) -> dict: #Init at -1,1

    #Initialise graph features
    subgraph_features = {}

    #Create an empty legacy features
    legacy_features = {}

    #Extract the key for the dictionary
    dict_keys_per_run = subgraph.keys()

    for key in dict_keys_per_run:
        if key in final_layer_dict:
            legacy_features[key] = final_layer_dict[key] #Reuse the features from the previous run
        else:
            subgraph_features[key] = subgraph[key][3] # Use the polarity as the key

    #Find the median of the legacy features
    median_legacy_features = np.median(np.array(list(legacy_features.values())))


    for legacy_key, legacy_value in legacy_features.items():
        if legacy_value < median_legacy_features:
            legacy_features[legacy_key] = lower_val #All values below the median are initialised as -1.
        else:
            legacy_features[legacy_key] = upper_val #All values above the median are initialised as 1.

    #Add the new features
    subgraph_features = legacy_features | subgraph_features

    return subgraph_features

def initialise_features_aegnn(subgraph: dict[int,tuple[float,float,float,int]], final_layer_legacy_dict: dict[int,np.float64]) -> tuple[dict, set]:

    new_features = set({})

    for event in subgraph.keys():

        if not event in final_layer_legacy_dict:
            new_features.add(event)

    return initialise_features_ver_1(subgraph, {}), new_features

##################################
####### Readout functions ########
##################################

#Create a proper histogram
def create_histogram(state_vector: np.ndarray, B: int, normalised_histograms: bool, min_max_normalisation: bool) -> np.ndarray:

    #Find the upper and the lower bound of the list
    lower_bound = np.min(state_vector)
    upper_bound = np.max(state_vector)

    #Create the bins
    bins = np.linspace(lower_bound, upper_bound, B+1)

    #Find the histogram
    histogram, _ = np.histogram(state_vector, bins=bins)

    if normalised_histograms:

        if min_max_normalisation:
            if histogram.max() - histogram.min() == 0:
                histogram = histogram/histogram.max()
            else:
                histogram = (histogram - histogram.min())/(histogram.max() - histogram.min())
        else:
            histogram = histogram/np.sum(state_vector)

    return histogram

def create_histogram_from_dict(state_vector_dict: dict, B: int, normalised_histograms: bool, min_max_normalisation: bool) -> np.ndarray:


    # Transform from dictionary to numpy vector
    state_vector_l = np.array(list(state_vector_dict.values()), dtype='f8')

    #Return the usual histogram
    return create_histogram(state_vector_l,B, normalised_histograms, min_max_normalisation)


# Create histogram, regarding the order of elements
def create_histogram_no_order(state_vector: np.ndarray) -> np.ndarray:

    #n = number of possible connections
    n = state_vector.shape[0]

    #The state vector is sorted, so that we can perform the count
    sorted_state_vector = np.sort(state_vector)

    #Initialise the state vector
    state_dist = np.zeros(n)

    #Record the first instance
    dist_index = 0
    state_dist[dist_index] = 1

    for i in range(1, n):

        #If the colours differ, then increment the index
        if sorted_state_vector[i] != sorted_state_vector[i - 1]:
            dist_index += 1

        #Add the value
        state_dist[dist_index] += 1

    return state_dist




##################################
####### Graph propagation ########
##################################
#Extract the node features from a subgraph
def get_node_features_final_layer_only(subgraph_adj_matrix_numpy: np.ndarray,
                                       initial_node_features: dict,
                                       iters: int = 5, B: int = 4) -> dict[int,]:

    #Create the initial state vector and histogram
    state_vector_l_prev = np.array(list(initial_node_features.values()), dtype='f8') #Use 64 bit float

    #Specify a maximum number of iterations
    for _ in range(iters):

        state_vector_l = layer_ver_3(subgraph_adj_matrix_numpy, state_vector_l_prev, B)

        #Change the new distribution state with the previous one
        state_vector_l_prev = state_vector_l

    #Create the histogram
    node_features_hist = create_histogram(state_vector_l_prev,B)

    list_node_keys = list(initial_node_features.keys())
    node_features = {list_node_keys[i]: state_vector_l[i] for i in range(len(list_node_keys))}

    return node_features_hist, node_features


#Extract all the node features from a subgraph
def get_node_features_all_layers(subgraph_adj_matrix_numpy: np.ndarray,
                                 initial_node_features: dict,
                                 iters: int, B: int,
                                 normalisation_type: str,
                                 apply_clipping: bool, reduce_min: bool) -> tuple[np.ndarray, dict]:

    node_feature_keys = list(initial_node_features.keys())

    #Create the initial state vector and histogram
    state_vector_l_prev = np.array(list(initial_node_features.values()), dtype='f8') #Use 64 bit float



    #Establish the normalisation style
    normalise_all_features_vector = min_max_normalisation = normalise_per_layer_feature_vector = min_max_per_layer_normalisation = False
    match normalisation_type:
        case ne.Normalisation.SIMPLE_NORM.value:
            normalise_all_features_vector = True
        case ne.Normalisation.SIMPLE_NORM_PER_LAYER.value:
            normalise_per_layer_feature_vector = True
        case ne.Normalisation.MIN_MAX_NORM.value:
            normalise_all_features_vector = min_max_normalisation = True
        case ne.Normalisation.MIN_MAX_NORM_PER_LAYER.value:
            normalise_per_layer_feature_vector = min_max_per_layer_normalisation = True
        case ne.Normalisation.NO_NORM.value:
            normalise_all_features_vector = min_max_normalisation = normalise_per_layer_feature_vector = min_max_per_layer_normalisation = False
        case _:
            raise Exception("Not a valid expression for normalisation")

    #Initialise the histogram, collecting all features
    feature_histogram = np.array([])

    #Specify a maximum number of iterations
    for _ in range(iters):

        state_vector_l = layer_ver_3(subgraph_adj_matrix_numpy, state_vector_l_prev, B,apply_clipping=apply_clipping,reduce_min=reduce_min)

        #Change the new distribution state with the previous one
        state_vector_l_prev = state_vector_l

        current_histogram = create_histogram(state_vector_l_prev,B,normalised_histograms=normalise_per_layer_feature_vector,min_max_normalisation=min_max_per_layer_normalisation)
        feature_histogram = np.concatenate((feature_histogram, current_histogram))


    feature_final_layer_dict = {node_feature_keys[i]: state_vector_l[i] for i in range(len(node_feature_keys))}


    if normalise_all_features_vector:

        if min_max_normalisation:
            if (feature_histogram.max() - feature_histogram.min()) == 0:
                feature_histogram = feature_histogram/feature_histogram.max()
            else:
                feature_histogram = (feature_histogram - feature_histogram.min())/(feature_histogram.max() - feature_histogram.min())
        else:
            feature_histogram = feature_histogram/np.sum(feature_histogram)

    return feature_histogram, feature_final_layer_dict

#Extract all the node features from a subgraph in an aegnn style
def get_node_features_all_layers_aegnn(subgraph_adj_source_target_numpy: np.ndarray, tracking_features: set,
                                       subgraph_features_keys: list[int], node_features_all_layers: list[dict],
                                       iters: int, B: int,
                                       normalisation_type: str,
                                       apply_clipping: bool, reduce_min: bool, alpha: float) -> tuple[np.ndarray, dict]:


    #Establish the normalisation style
    normalise_all_features_vector = min_max_normalisation = normalise_per_layer_feature_vector = min_max_per_layer_normalisation = False
    match normalisation_type:
        case ne.Normalisation.SIMPLE_NORM.value:
            normalise_all_features_vector = True
        case ne.Normalisation.SIMPLE_NORM_PER_LAYER.value:
            normalise_per_layer_feature_vector = True
        case ne.Normalisation.MIN_MAX_NORM.value:
            normalise_all_features_vector = min_max_normalisation = True
        case ne.Normalisation.MIN_MAX_NORM_PER_LAYER.value:
            normalise_per_layer_feature_vector = min_max_per_layer_normalisation = True
        case ne.Normalisation.NO_NORM.value:
            normalise_all_features_vector = min_max_normalisation = normalise_per_layer_feature_vector = min_max_per_layer_normalisation = False
        case _:
            raise Exception("Not a valid expression for normalisation")


    #Initialise the feature histogram
    feature_histogram = np.array([])

    #iters = maximum number of iterations
    for l in range(iters):

        #Obtain all the possible features from this layer
        features_current_layer = node_features_all_layers[l]

        #Extract only the ones, that represent the current graph
        features_current_layer = {node: features_current_layer[node] for node in subgraph_features_keys}

        #This dictionary contains all the events that are either new or are precessed in the previous layers
        past_features = {node_element for node_element in tracking_features}


        subgraph_node_features_l = None
        if alpha > 0:
            subgraph_node_features_l, tracking_features = layer_ver_aegnn_v2(subgraph_adj_source_target_numpy, node_features=features_current_layer,
                                                                             past_nodes=tracking_features, alpha=alpha)
        else:
            subgraph_node_features_l, tracking_features = layer_ver_aegnn(subgraph_adj_source_target_numpy, node_features=features_current_layer,
                                                                      past_nodes=tracking_features)


        updated_node_features_l = {}
        for node, node_feature in subgraph_node_features_l.items():
            if node in past_features:
                updated_node_features_l[node] = node_feature
            else:
                updated_node_features_l[node] = node_features_all_layers[l+1][node]


        current_histogram = create_histogram_from_dict(updated_node_features_l, B,normalised_histograms=normalise_per_layer_feature_vector,min_max_normalisation=min_max_per_layer_normalisation)
        feature_histogram = np.concatenate((feature_histogram, current_histogram))


        #Add the new features
        node_features_all_layers[l+1] |= updated_node_features_l


    if normalise_all_features_vector:
        if min_max_normalisation:
            if (feature_histogram.max() - feature_histogram.min()) == 0:
                feature_histogram = feature_histogram/feature_histogram.max()
            else:
                feature_histogram = (feature_histogram - feature_histogram.min())/(feature_histogram.max() - feature_histogram.min())
        else:
            feature_histogram = feature_histogram/np.sum(feature_histogram)


    return feature_histogram, node_features_all_layers


##################################
######## Features Readout ########
##################################

#The function executes the neuromorphic step of the graph.
def readout_graph_features(events: dict[int, tuple[float, float, float, float]],
                           time_steps: list[float], delta_t: float,
                           num_layers: int, beta: float,
                           B: int,
                           normalisation_type: str,
                           apply_clipping: bool, reduce_min: bool,
                           legacy_features_init_type: str) -> np.ndarray:

    #Initialise node features
    node_features = []

    final_layer_dict = {}

    #Iterate through all the possible windows
    for t in time_steps:

        ###### Create graph ######

        #Initialise the nodes, and hence the graph
        subgraph_in_time_window = slide_window(events, t, delta_t) #Creates the vertices

        #Obtain the adjacency matrix in an np.ndarray form
        subgraph_adj_matrix_numpy = tools.create_adjacency_matrix_3D(events=subgraph_in_time_window, time_weight=beta) #Create the edges

        #### Create graph (End)###


        #Initialise the features for this subgraph
        subgraph_features = {}
        match legacy_features_init_type:
            case 'default':
                subgraph_features = initialise_features_ver_1(subgraph_in_time_window, final_layer_dict)
            case 'last_layer':
                subgraph_features = initialise_features_ver_2(subgraph_in_time_window, final_layer_dict)
            case 'binary':
                subgraph_features = initialise_features_ver_3(subgraph_in_time_window, final_layer_dict)

        #Compute the graph features for this step
        subgraph_node_features, final_layer_features = get_node_features_all_layers(subgraph_adj_matrix_numpy,
                                                                                    iters=num_layers, initial_node_features=subgraph_features,
                                                                                    B=B, normalisation_type=normalisation_type,
                                                                                    apply_clipping=apply_clipping, reduce_min=reduce_min)

        final_layer_dict |= final_layer_features

        #Add the new features to the graph/update the existing features
        node_features.append(subgraph_node_features)


    return node_features



#The function executes the neuromorphic step of the graph, based on the AEGNN algorithm.
def readout_graph_features_aegnn(events: dict[int, tuple[float, float, float, float]] | list[tuple[float, float, float, float]],
                                 time_steps: list[float], delta_t: float,
                                 num_layers: int, beta: float,
                                 B: int,
                                 normalisation_type: str,
                                 apply_clipping: bool, reduce_min: bool,
                                 subsample_spike_per_pixel: bool, alpha: float) -> np.ndarray:

    #If the events are specified as a list, convert it to a dictionary
    if isinstance(events, list):
        events = tools.convert_to_dict(events)


    #Initialise node features that are used for the kernel
    features_histogram = []

    #The node features, collecting the node features for each layer.
    node_features_all_layers = [{} for _ in range(num_layers + 1)] # This should be a list of dictionaries. Each should represent one layer

    #Iterate through all the possible windows
    for t in time_steps:

        ###### Create graph ######

        #Initialise the nodes, and hence the graph
        subgraph_in_time_window = slide_window(events, t, delta_t) #Creates the vertices

        #We check if there are features in the current time window. If not, we assign them null
        if subgraph_in_time_window:

            coordinates_events = None #Necessary only for subsampling one spike per pixel
            if subsample_spike_per_pixel:
                subgraph_in_time_window, coordinates_events = tools.subsample_spike_per_pixel(subgraph_in_time_window)

            #Note: The edges are initialised as a 2D list, with row entires = 2. It is a list from source to target.
            subgraph_adj_source_target_numpy = tools.create_adjacency_list_3D(events=subgraph_in_time_window, time_weight=beta) #Create the edges

            #### Create graph (End)###

            #Initialise features as AEGNN (It is the same principle behind version 1, but returns only the new nodes, to be included eventually)
            subgraph_features, new_features_set = initialise_features_aegnn(subgraph_in_time_window, node_features_all_layers[-1])

            #Add the new node features to the first layer only
            node_features_all_layers[0] |= subgraph_features

            #Obtain the keys for the current graph, used to find the past nodes
            subgraph_features_keys = list(subgraph_features.keys())

            #Compute the graph features for this step
            subgraph_features_histogram, node_features_all_layers = get_node_features_all_layers_aegnn(subgraph_adj_source_target_numpy, tracking_features=new_features_set,
                                                                                                       iters=num_layers,
                                                                                                       subgraph_features_keys=subgraph_features_keys, node_features_all_layers=node_features_all_layers,
                                                                                                       B=B, normalisation_type=normalisation_type,
                                                                                                       apply_clipping=apply_clipping, reduce_min=reduce_min, alpha=alpha)

            if subsample_spike_per_pixel:
                #Collect the subgraph again
                node_features_all_layers = tools.expand_spike_per_pixel(subgraph_in_time_window, node_features_all_layers, coordinates_events, convert_key_to_string=False)

        else: #If no features exists, then we create an empty vector
            subgraph_features_histogram = np.zeros(B*num_layers)


        #Append the information to the node features, fed into the kernel
        features_histogram.append(subgraph_features_histogram)

    return features_histogram

#Manually create the Weisfelier-Lehman coordinates.
#The procedure described in https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/
def obtain_dataset_neuromorphic_gnn_features(graphs_dataset: list[list[tuple[float,float,float]]], time_steps: list[float], delta_t: float,
                                             num_layers: int, beta: float, B:int, normalisation_type: str, apply_clipping: bool, reduce_min: bool,
                                             legacy_features_init_type: str, subsample_spike_per_pixel: bool, alpha: float) -> tuple[list[np.ndarray], list[int]]:


    #Create the empty datasets
    neuromorphic_features = []
    label_dataset = []

    for graph_coords, label in graphs_dataset:


        if legacy_features_init_type == 'aegnn':
            graph_features = readout_graph_features_aegnn(graph_coords, time_steps, delta_t,
                                                          num_layers=num_layers,
                                                          beta=beta, B=B,
                                                          normalisation_type=normalisation_type,
                                                          apply_clipping=apply_clipping, reduce_min=reduce_min,
                                                          subsample_spike_per_pixel=subsample_spike_per_pixel, alpha=alpha)
        else:

            graph_features = readout_graph_features(graph_coords, time_steps, delta_t,
                                                    num_layers=num_layers,
                                                    beta=beta, B=B,
                                                    normalisation_type=normalisation_type,
                                                    apply_clipping=apply_clipping, reduce_min=reduce_min,
                                                    legacy_features_init_type=legacy_features_init_type)

        #Append the values to the dataset
        neuromorphic_features.append(graph_features)   #An NxM 2D array, where N = number of graphs to quantify and M = number of features
        label_dataset.append(label)

    return neuromorphic_features, label_dataset

#####################################################################################
################################## Main Program #####################################
#####################################################################################


def execute_kernel_run_optimise_layer(graph_dataset: list[list[tuple[float,float,float]]],
                       time_steps: list[float], delta_t: float,
                       kernel_function, error_func,
                       num_layers_list: list[int],
                       beta: float = 1.0,
                       B: int = 10,
                       normalisation_type: str = ne.Normalisation.MIN_MAX_NORM_PER_LAYER.value,
                       apply_clipping: bool = False, reduce_min: bool = False,
                       legacy_features_init_type: str = 'default'):


    optimal_val_score = 0.0

    optimal_classifier_vec = None
    optimal_training_score = 0.0

    optimal_input_train_data_labels = None
    optimal_input_val_data_labels = None
    optimal_input_test_data_labels = None

    optimal_layers = 0
    for num_layers in num_layers_list:

        print(f"num_layers: {num_layers}")
        #Obtain the final feature vector to extract the coordinates
        state_vec_dataset, label_dataset = obtain_dataset_neuromorphic_gnn_features(graph_dataset, time_steps, delta_t, num_layers=num_layers, beta=beta, B=B,
                                                                                    normalisation_type=normalisation_type, apply_clipping=apply_clipping, reduce_min=reduce_min,
                                                                                    legacy_features_init_type=legacy_features_init_type)


        n = len(time_steps)

        #Finally, implement the solution as via the paper.
        classifier_vec, score_results, input_train_data_labels, input_val_data_labels, input_test_data_labels, _, _ = krn.run_kernel_val_set_majority_rule(state_vec_dataset, label_dataset, kernel_function, error_func, n)

        score_results_train, score_results_val, _ = score_results


        print(f"The score result for the training set using {num_layers} layers is {score_results_train}")
        print(f"The score result for the validation set using {num_layers} layers is {score_results_val}")
        if optimal_val_score < score_results_val:

            optimal_classifier_vec = classifier_vec
            optimal_training_score = score_results_train


            optimal_input_train_data_labels = input_train_data_labels
            optimal_input_val_data_labels = input_val_data_labels
            optimal_input_test_data_labels = input_test_data_labels

            optimal_val_score = score_results_val
            optimal_layers = num_layers

    print(f"The optimal number of layers is {optimal_layers}")
    print(f"The optimal validation score, using majority rule is {optimal_val_score}")
    return optimal_classifier_vec, (optimal_training_score, optimal_val_score), optimal_input_train_data_labels, optimal_input_val_data_labels, optimal_input_test_data_labels, optimal_layers


def execute_kernel_run_optimise_layer_using_testset(graph_dataset: list[list[tuple[float,float,float]]],
                                                    time_steps: list[float], delta_t: float,
                                                    kernel_function, error_func,
                                                    num_layers_list: list[int],
                                                    beta: float = 1.0,
                                                    B: int = 10,
                                                    normalisation_type: str = ne.Normalisation.MIN_MAX_NORM_PER_LAYER.value,
                                                    apply_clipping: bool = False, reduce_min: bool = False,
                                                    legacy_features_init_type: str = 'default'):

    optimal_val_score = 0.0
    optimal_test_score = 0.0

    optimal_classifier_vec = None
    optimal_training_score = 0.0

    optimal_input_train_data_labels = None
    optimal_input_val_data_labels = None
    optimal_input_test_data_labels = None

    optimal_layers = 0
    for num_layers in num_layers_list:

        print(f"num_layers: {num_layers}")
        #Obtain the final feature vector to extract the coordinates
        state_vec_dataset, label_dataset = obtain_dataset_neuromorphic_gnn_features(graph_dataset, time_steps, delta_t, num_layers=num_layers, beta=beta, B=B,
                                                                                    normalisation_type=normalisation_type, apply_clipping=apply_clipping, reduce_min=reduce_min,
                                                                                    legacy_features_init_type=legacy_features_init_type)


        n = len(time_steps)

        #Finally, implement the solution as via the paper.
        classifier_vec, score_results, input_train_data_labels, input_val_data_labels, input_test_data_labels, _, _ = krn.run_kernel_val_set_majority_rule(state_vec_dataset, label_dataset, kernel_function, error_func, n, train_validation_cutoff=0.99)

        score_results_train, score_results_val, score_results_test = score_results


        print(f"The score result for the training set using {num_layers} layers is {score_results_train}")
        print(f"The score result for the validation set using {num_layers} layers is {score_results_val}")
        print(f"The score result for the testing set using {num_layers} layers is {score_results_test}")
        if optimal_test_score < score_results_test:

            optimal_classifier_vec = classifier_vec
            optimal_training_score = score_results_train


            optimal_input_train_data_labels = input_train_data_labels
            optimal_input_val_data_labels = input_val_data_labels
            optimal_input_test_data_labels = input_test_data_labels

            optimal_val_score = score_results_val

            optimal_test_score = score_results_test
            optimal_layers = num_layers

    print(f"The optimal number of layers is {optimal_layers}")
    print(f"The optimal validation score, using majority rule is {optimal_test_score}")
    return optimal_classifier_vec, (optimal_training_score, optimal_val_score, optimal_test_score), optimal_input_train_data_labels, optimal_input_val_data_labels, optimal_input_test_data_labels




def execute_kernel_run(graph_dataset: list[list[tuple[float,float,float]]],
                       time_steps: list[float], delta_t: float,
                       kernel_function, error_func,
                       num_layers: int = 4, beta: float = 1.0,
                       B: int = 10,
                       normalisation_type: str = ne.Normalisation.MIN_MAX_NORM_PER_LAYER.value,
                       apply_clipping: bool = False, reduce_min: bool = False,
                       legacy_features_init_type: str = 'default',
                       train_test_cutoff: float = 0.8,
                       subsample_spike_per_pixel: bool = False,
                       alpha: float = 0.0):


    #Obtain the final feature vector to extract the coordinates
    state_vec_dataset, label_dataset = obtain_dataset_neuromorphic_gnn_features(graph_dataset, time_steps, delta_t, num_layers=num_layers, beta=beta, B=B,
                                                                                normalisation_type=normalisation_type, apply_clipping=apply_clipping, reduce_min=reduce_min,
                                                                                legacy_features_init_type=legacy_features_init_type, subsample_spike_per_pixel=subsample_spike_per_pixel, alpha=alpha)

    n = len(time_steps)

    #Finally, implement the solution as via the paper.
    classifier_vec, score_results, (input_train_dataset, labels_train), (input_test_dataset, labels_test), _ = krn.run_kernel_majority_rule(state_vec_dataset, label_dataset, kernel_function, error_func, n, train_test_cutoff=train_test_cutoff)

    return classifier_vec, score_results, (input_train_dataset, labels_train), (input_test_dataset, labels_test)



def execute_kernel_run_v2(graph_dataset: list[list[tuple[float,float,float]]],
                          time_steps: list[float], delta_t: float,
                          kernel_function, error_func,
                          num_layers: int = 4, beta: float = 1.0,
                          B: int = 10,
                          normalisation_type: str = ne.Normalisation.MIN_MAX_NORM_PER_LAYER.value,
                          apply_clipping: bool = False, reduce_min: bool = False,
                          legacy_features_init_type: str = 'aegnn',
                          train_test_cutoff: float = 0.8,
                          #DEPRICATED for this function: subsample_spike_per_pixel: bool = False,
                          alphas: list[float] = [1.0]):


    #Create the empty datasets
    #graphs_all_times = []
    label_dataset = []

    num_graphs = len(graph_dataset)


    #Create the classifier vector, containing the best classifier for all runs
    classifier_vec = []

    #Initialise the optimal quantum evolution time array
    alpha_optimal_list = []


    #Initilise the training input dataset for all graph evolutions
    input_train_dataset_all_graph_time = []
    labels_train = []

    #Initilise the test input dataset for all graph evolutions
    input_test_dataset_all_graph_time = []
    labels_test = []

    if legacy_features_init_type == 'aegnn':


        #The node features, collecting the node features for each layer.
        node_features_all_layers = [[{} for _ in range(num_layers + 1)] for _ in range(num_graphs)] # This should be a list of dictionaries. Each should represent one layer

        ##Iterate through all the possible windows
        for t in time_steps:


            node_features_all_alphas_all_graphs = []

            #Initialise node features that are used for the kernel
            graphs_histograms = []

            label_dataset = []
            for i, (events, label) in enumerate(graph_dataset):

                #If the events are specified as a list, convert it to a dictionary
                if isinstance(events, list):
                    events = tools.convert_to_dict(events)

                ###### Create graph ######

                #Initialise the nodes, and hence the graph
                subgraph_in_time_window = slide_window(events, t, delta_t) #Creates the vertices

                #We check if there are features in the current time window. If not, we assign them null
                if subgraph_in_time_window:

                    #Note: The edges are initialised as a 2D list, with row entires = 2. It is a list from source to target.
                    subgraph_adj_source_target_numpy = tools.create_adjacency_list_3D(events=subgraph_in_time_window, time_weight=beta) #Create the edges

                    #### Create graph (End)###


                    #Initialise features as AEGNN (It is the same principle behind version 1, but returns only the new nodes, to be included eventually)
                    subgraph_features, new_features_set = initialise_features_aegnn(subgraph_in_time_window, node_features_all_layers[i][-1])


                    #Add the new node features to the first layer only
                    node_features_all_layers[i][0] |= subgraph_features


                    #Obtain the keys for the current graph, used to find the past nodes
                    subgraph_features_keys = list(subgraph_features.keys())


                    node_features_all_alpha = []
                    subgraph_features_histogram_all_alphas = []
                    for alpha in alphas:

                        #Compute the graph features for this step
                        subgraph_features_histogram, node_features_one_alpha = get_node_features_all_layers_aegnn(subgraph_adj_source_target_numpy, tracking_features=new_features_set,
                                                                                                                  iters=num_layers,
                                                                                                                  subgraph_features_keys=subgraph_features_keys, node_features_all_layers=node_features_all_layers[i],
                                                                                                                  B=B, normalisation_type=normalisation_type,
                                                                                                                  apply_clipping=apply_clipping, reduce_min=reduce_min, alpha=alpha)


                        subgraph_features_histogram_all_alphas.append(subgraph_features_histogram)
                        node_features_all_alpha.append(node_features_one_alpha)

                    node_features_all_alphas_all_graphs.append(node_features_all_alpha) #[[[{} for _ in range(num_layers + 1)] for _ in range(len([alphas]))] for _ in range(num_graphs)] # This should be a list of dictionaries. Each should represent one layer




                else: #If no features exists, then we create an empty vector
                    if len(alphas) == 1:
                        subgraph_features_histogram_all_alphas = np.zeros(B*num_layers)
                    else:
                        subgraph_features_histogram_all_alphas = [np.zeros(B*num_layers)]*len(alphas)



                #Append the information to the node features, fed into the kernel
                graphs_histograms.append(subgraph_features_histogram_all_alphas)


                label_dataset.append(label)

            #Classify the dataset

            #Finally, find the optimal alpha
            classifier, _, train_dataset, test_dataset, predicted_labels_train, optimal_alpha, alpha_index = krn.run_kernel_optimal_alpha(graphs_histograms, label_dataset,
                                                                                                                                          kernel_function, error_func,
                                                                                                                                          alphas,
                                                                                                                                          train_test_cutoff)


            #Obtain the input and the labels from the dataset
            input_train_dataset, labels_train = train_dataset #Notice that the labels shouldn't change
            input_test_dataset, labels_test = test_dataset



            #Obtain the optimal time initial features
            optimal_node_features = [node_features[alpha_index] for node_features in node_features_all_alphas_all_graphs]



            #Update the individual dictionaries
            for i in range(len(optimal_node_features)):
                for l in range(num_layers + 1):
                    node_features_all_layers[i][l] |= optimal_node_features[i][l]


            #Save the optimal parameters
            classifier_vec.append(classifier)

            #Append the optimal quantum time
            alpha_optimal_list.append(optimal_alpha)


            #Append the dataset information
            #Training dataset
            input_train_dataset_all_graph_time.append(input_train_dataset)

            #Testing dataset
            input_test_dataset_all_graph_time.append(input_test_dataset)


            if t == 0:
                vote_array = predicted_labels_train
            else:
                vote_array = np.vstack((vote_array, predicted_labels_train))


            #Append the values to the dataset
            #graphs_all_times.append(graphs_histograms)   #An NxM 2D array, where N = number of graphs to quantify and M = number of features


    else:
        raise Exception("This is not implemented and not allowed in this case")

    final_prediction = [tools.count_vote(vote_array[:,i]) for i in range(len(input_train_dataset_all_graph_time[0]))]

    score_results = error_func(final_prediction,np.array(labels_train))

    return classifier_vec, score_results, (input_train_dataset_all_graph_time, labels_train), (input_test_dataset_all_graph_time, labels_test), alpha_optimal_list
