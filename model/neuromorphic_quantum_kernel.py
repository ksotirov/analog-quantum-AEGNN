####### External libraries #######
import numpy as np

####### Defined libraries ########
from . import kernel as krn
from . import tools
from .process_quantum_graph import QuantumGraph


###########################################
############# Dataset Process #############
###########################################


#a) Firstly we extract the coordinates of every graph
#Gets the features needed for the dataset.
def obtain_dataset_neuromorphic_gnn_features_sequential(graphs_dataset: list[list[tuple[float,float,float]]], time: list[float] | float, delta_t: float,
                                                        quantum_evolution_time: list[int],
                                                        recenter: bool, device_type: str,
                                                        legacy_features_init_type: str,
                                                        one_window: bool,
                                                        final_layer_dict: list[dict[int,np.float64]] = []) -> tuple[list[np.ndarray], list[int]]:

    #Create the empty datasets
    neuromorphic_features = []
    label_dataset = []
    graph_qtime_features = [] #This is only necessary if one_window = True

    qgraph = QuantumGraph(time=time, delta_t=delta_t, quantum_evolution_time=quantum_evolution_time,
                          recenter=recenter, device_type=device_type, legacy_features_init_type=legacy_features_init_type, one_graph_window=one_window)

    for i, (graph_coords, label) in enumerate(graphs_dataset):


        if one_window:
            #Essentially graph_features is a NxM vector:
            # N => different quantum evolution times and
            # M => different features
            graph_features, graph_features_all_qtime = qgraph.readout_graph_features(graph_coords, final_layer_dict=final_layer_dict[i])

            graph_qtime_features.append(graph_features_all_qtime) #NxQT, where N = num_graphs, QT = quantum time evolution
        else:
            #Essentially graph_features is a TxNxM vector:
            # T => the different time windows for the execution,
            # N => different quantum evolution times and
            # M => different features
            graph_features = qgraph.readout_graph_features(graph_coords)

        #Append the values to the dataset
        neuromorphic_features.append(graph_features)
        label_dataset.append(label)

    return neuromorphic_features, label_dataset, graph_qtime_features




###########################################
############### One window ################
###########################################

#a) Firstly we extract the coordinates of every graph
#Gets the features needed for the dataset.

def obtain_dataset_neuromorphic_gnn_features_sequential_one_window(graphs_dataset: list[list[tuple[float,float,float]]], time: float, delta_t: float,
                                                                   final_layer_dict: list[dict[int,np.float64]],
                                                                   quantum_evolution_time: list[int],
                                                                   recenter: bool, device_type: str,
                                                                   legacy_features_init_type: str,
                                                                   dimensions: int, normalise_xy: bool,
                                                                   subsample_spike_per_pixel: bool,
                                                                   linear_multiplicative_time_c: float,
                                                                   split_time_in_middle: bool,
                                                                   number_of_quantum_shots: int,
                                                                   remove_close_points: bool,
                                                                   allow_noise: bool, noise_model_params,
                                                                   detuning_encoding: bool,
                                                                   max_train_event: int,
                                                                   cutoff_val: int) -> tuple[list[np.ndarray], list[int]]:

    #Create the empty datasets
    neuromorphic_features = []
    label_dataset = []
    graph_qtime_features = []


    #Find the max graph
    max_num_events_per_t = tools.find_max_number_of_events(graphs_dataset, time, delta_t)
    if max_train_event > 0:
        max_num_events_per_t = max(max_num_events_per_t, max_train_event)

    qgraph = QuantumGraph(time=time, delta_t=delta_t, quantum_evolution_time=quantum_evolution_time,
                          recenter=recenter, device_type=device_type, legacy_features_init_type=legacy_features_init_type,
                          one_graph_window=True, dimensions=dimensions,
                          normalise_xy=normalise_xy, max_num_events=max_num_events_per_t,
                          subsample_spike_per_pixel=subsample_spike_per_pixel,
                          linear_multiplicative_time_c=linear_multiplicative_time_c,split_time_in_middle=split_time_in_middle,
                          number_of_quantum_shots=number_of_quantum_shots,
                          remove_close_points=remove_close_points,
                          allow_noise=allow_noise, noise_model_params=noise_model_params,
                          detuning_encoding=detuning_encoding)


    for i, (graph_coords, label) in enumerate(graphs_dataset):


        if cutoff_val > 0 and i >= cutoff_val:
            subgraph_features = []
            for _ in range(len(quantum_evolution_time)):
                p_dist_per_time = np.zeros(max_num_events_per_t + 1)
                p_dist_per_time[0] = 1 #Fix error!
                subgraph_features.append(p_dist_per_time)

            graph_features_all_qtime = [{} for _ in range(len(quantum_evolution_time))]
        else:
            #break # No need to process the graph

            #Essentially graph_features is a NxM vector:
            # N => different quantum evolution times and
            # M => different features
            subgraph_features, graph_features_all_qtime = qgraph.readout_graph_features(graph_coords, final_layer_dict=final_layer_dict[i])

        #Append the values to the dataset
        neuromorphic_features.append(subgraph_features)
        label_dataset.append(label)
        graph_qtime_features.append(graph_features_all_qtime) #NxQT, where N = num_graphs, QT = quantum time evolution

    return neuromorphic_features, label_dataset, graph_qtime_features, max_num_events_per_t


#####################################################################################
################################## Main Program #####################################
#####################################################################################


#The program that executes the entire run.
def execute_kernel_run(graph_dataset: list[list[tuple[float,float,float]]],
                       time_steps: list[float], delta_t: float,
                       kernel_function, error_func,
                       q_evolution_min_t: int, q_evolution_max_t: int, q_evolution_delta_t: int,
                       recenter: bool = True, device_type: str = 'analog',
                       legacy_features_init_type: str = 'default'
                       ):


    #Create the time range of quantum evolution.
    quantum_evolution_time_list = list(range(q_evolution_max_t, q_evolution_min_t, -q_evolution_delta_t))

    ### 1) First features are extracted
    state_vec_dataset, label_dataset, _ = obtain_dataset_neuromorphic_gnn_features_sequential(graph_dataset, time_steps, delta_t,
                                                                                              quantum_evolution_time_list, recenter,
                                                                                              device_type, legacy_features_init_type)

    n = len(time_steps)

    #Finally, implement the solution as via the paper.
    classifier_vec, score_results, (input_train_dataset, labels_train), (input_test_dataset, labels_test), _, qevol_time_list = krn.run_kernel_majority_rule_quantum(state_vec_dataset, label_dataset, kernel_function, error_func, n, quantum_evolution_time_list)

    return classifier_vec, score_results, (input_train_dataset, labels_train), (input_test_dataset, labels_test), qevol_time_list



#The program that executes the entire run.
#In the second version, we attempt to pick the best possible time, in order to initialise the qubits
def execute_kernel_run_v2(graph_dataset: list[list[tuple[float,float,float]]],
                          time_steps: list[float], delta_t: float,
                          kernel_function, error_func,
                          q_evolution_min_t: int, q_evolution_max_t: int, q_evolution_delta_t: int,
                          recenter: bool = True, device_type: str = 'analog',
                          legacy_features_init_type: str = 'default',
                          dimensions: int = 2, normalise_xy: bool = False,
                          subsample_spike_per_pixel: bool = False,
                          train_test_ratio: float = 0.8,
                          linear_multiplicative_time_c: float = 0.0,
                          split_time_in_middle: bool = False,
                          number_of_quantum_shots: int = 100,
                          remove_close_points: bool = False,
                          allow_noise: bool = False, noise_model_params = None,
                          save_q_evol_time: str = '',
                          detuning_encoding: bool = False,
                          return_max_num_events: bool = False):


    #Create the vote array
    vote_array = np.array([])

    #Create the classifier vector, containing the best classifier for all runs
    classifier_vec = []

    #Initialise the optimal quantum evolution time array
    qevolution_optimal_time_list = []

    #Initilise the training input dataset for all graph evolutions
    input_train_dataset_all_graph_time = []
    labels_train = []

    #Initilise the test input dataset for all graph evolutions
    input_test_dataset_all_graph_time = []
    labels_test = []

    #Create the time range of quantum evolution. (Increasing order)
    quantum_evolution_time_list = list(range(q_evolution_min_t, q_evolution_max_t, q_evolution_delta_t))

    #Initialise the dictionaries of features in the optimal run
    optimal_run_feature_list_of_dict = [{} for _ in range(len(graph_dataset))]


    max_events_list = []
    for t in time_steps:

        cutoff_val = 0
        if return_max_num_events:
            cutoff_val = int(len(graph_dataset)*train_test_ratio)
        state_vec_dataset, label_dataset, graph_all_qtime_features, max_events = obtain_dataset_neuromorphic_gnn_features_sequential_one_window(graph_dataset, t, delta_t,
                                                                                                                                optimal_run_feature_list_of_dict,
                                                                                                                                quantum_evolution_time_list, recenter,
                                                                                                                                device_type, legacy_features_init_type,
                                                                                                                                dimensions, normalise_xy, subsample_spike_per_pixel,
                                                                                                                                linear_multiplicative_time_c=linear_multiplicative_time_c,
                                                                                                                                split_time_in_middle=split_time_in_middle,
                                                                                                                                number_of_quantum_shots=number_of_quantum_shots,
                                                                                                                                remove_close_points=remove_close_points,
                                                                                                                                allow_noise=allow_noise, noise_model_params=noise_model_params,
                                                                                                                                detuning_encoding=detuning_encoding, max_train_event=0,
                                                                                                                                cutoff_val=cutoff_val)
        if return_max_num_events:
            max_events_list.append(max_events)


        final_str_to_save = ''
        if save_q_evol_time:
            final_str_to_save = save_q_evol_time + '_' + str(t) + '.csv'

        #Finally, run the quantum kernel.
        classifier, _, train_dataset, test_dataset, predicted_labels_train, qevol_time, qevol_index = krn.run_kernel_quantum(state_vec_dataset, label_dataset,
                                                                                                                             kernel_function, error_func,
                                                                                                                             quantum_evolution_time_list,
                                                                                                                             train_test_ratio, final_str_to_save)

        #Obtain the input and the labels from the dataset
        input_train_dataset, labels_train = train_dataset #Notice that the labels shouldn't change
        input_test_dataset, labels_test = test_dataset



        #Obtain the optimal time initial features
        optimal_qtime_features = [qtime_feature[qevol_index] for qtime_feature in graph_all_qtime_features]


        #Update the individual dictionaries
        for i in range(len(optimal_qtime_features)):
            optimal_run_feature_list_of_dict[i] |= optimal_qtime_features[i]


        #Save the optimal parameters
        classifier_vec.append(classifier)

        #Append the optimal quantum time
        qevolution_optimal_time_list.append(qevol_time)

        #Append the dataset information
        #Training dataset
        input_train_dataset_all_graph_time.append(input_train_dataset)

        #Testing dataset
        input_test_dataset_all_graph_time.append(input_test_dataset)


        if t == 0:
            vote_array = predicted_labels_train
        else:
            vote_array = np.vstack((vote_array, predicted_labels_train))


    final_prediction = [tools.count_vote(vote_array[:,i]) for i in range(len(input_train_dataset_all_graph_time[0]))]

    score_results = error_func(final_prediction,np.array(labels_train))

    if return_max_num_events:
        return classifier_vec, score_results, (input_train_dataset_all_graph_time, labels_train), (input_test_dataset_all_graph_time, labels_test), qevolution_optimal_time_list, max_events_list
    return classifier_vec, score_results, (input_train_dataset_all_graph_time, labels_train), (input_test_dataset_all_graph_time, labels_test), qevolution_optimal_time_list


def execute_kernel_run_v2_inference(test_graph_dataset: list[list[tuple[float,float,float]]],
                                    trained_classifiers,
                                    time_steps: list[float], delta_t: float,
                                    error_func,
                                    optimal_q_evolution_time_list: list[float],
                                    recenter: bool = True, device_type: str = 'analog',
                                    legacy_features_init_type: str = 'default',
                                    dimensions: int = 2, normalise_xy: bool = False,
                                    subsample_spike_per_pixel: bool = False,
                                    linear_multiplicative_time_c: float = 0.0,
                                    split_time_in_middle: bool = False,
                                    number_of_quantum_shots: int = 100,
                                    remove_close_points: bool = False,
                                    allow_noise: bool = False, noise_model_params = None,
                                    detuning_encoding: bool = False,
                                    max_train_events_vector: bool = []):


    #Create the vote array
    vote_array = np.array([])

    #Initilise the test input dataset for all graph evolutions
    input_test_dataset_all_graph_time = []
    labels_test = [label for _, label in test_graph_dataset]

    majority_rule_score_results = []
    per_time_window_score = []

    if not max_train_events_vector:
        max_train_events_vector = [0]*len(time_steps)

    #Initialise the dictionaries of features in the optimal run
    optimal_run_feature_list_of_dict = [{} for _ in range(len(test_graph_dataset))]

    for i, t in enumerate(time_steps):


        #Create the time range of quantum evolution. (Increasing order)
        quantum_evolution_time_list = list(range(optimal_q_evolution_time_list[i],2001, 50))# optimal_q_evolution_time_list[i] + 25, 50))

        ### 1) First features are extracted
        state_vec_dataset_test_all, _, graph_test_qtime_features, _ = obtain_dataset_neuromorphic_gnn_features_sequential_one_window(test_graph_dataset, t, delta_t,
                                                                                                                                     optimal_run_feature_list_of_dict,
                                                                                                                                     quantum_evolution_time_list, recenter,
                                                                                                                                     device_type, legacy_features_init_type,
                                                                                                                                     dimensions, normalise_xy, subsample_spike_per_pixel,
                                                                                                                                     linear_multiplicative_time_c=linear_multiplicative_time_c,
                                                                                                                                     split_time_in_middle=split_time_in_middle,
                                                                                                                                     number_of_quantum_shots=number_of_quantum_shots,
                                                                                                                                     remove_close_points=remove_close_points,
                                                                                                                                     allow_noise=allow_noise, noise_model_params=noise_model_params,
                                                                                                                                     detuning_encoding=detuning_encoding,
                                                                                                                                     max_train_event=max_train_events_vector[i],
                                                                                                                                     cutoff_val=0)



        #Obtain the first feature in this case
        state_vec_dataset_test = [state_vec[0] for state_vec in state_vec_dataset_test_all]

        #Obtain the optimal time initial features
        optimal_qtime_features = [qtime_feature[0] for qtime_feature in graph_test_qtime_features]# Only the first


        #Update the individual dictionaries
        for j in range(len(optimal_qtime_features)):
            optimal_run_feature_list_of_dict[j] |= optimal_qtime_features[j]


        # Make a prediction on the dataset in the current window
        test_prediction_per_window = krn.single_window_predict(state_vec_dataset_test, trained_classifiers[i])
        score_per_window = error_func(test_prediction_per_window, np.array(labels_test))
        per_time_window_score.append(score_per_window)

        #Testing dataset
        input_test_dataset_all_graph_time.append(state_vec_dataset_test)

        final_prediction = []
        if t == 0:
            vote_array = test_prediction_per_window
            final_prediction = [tools.count_vote(vote_array[j]) for j in range(len(input_test_dataset_all_graph_time[0]))]
        else:
            vote_array = np.vstack((vote_array, test_prediction_per_window))
            final_prediction = [tools.count_vote(vote_array[:,j]) for j in range(len(input_test_dataset_all_graph_time[0]))]


        #Produce accuracy score based on all previous windows

        score_results = error_func(final_prediction,np.array(labels_test))
        majority_rule_score_results.append(score_results)

    return (input_test_dataset_all_graph_time, labels_test),  majority_rule_score_results, per_time_window_score