#Import the necessary libraries
import numpy as np
import warnings
import time
import csv
import collections
from typing import Union

#Import ML libraries
from sklearn import svm
from scipy.spatial.distance import cdist

#Import quantum evolution libraries
import qutip
import pulser
import pulser_simulation
from pulser.noise_model import NoiseModel
import logging

#Import emulator libraries
import emu_mps
import emu_sv
from emu_mps.solver import Solver

##################################
### Setting up the observables ###
##################################

NON_LINDBLADIAN_NOISE = {"SPAM", "doppler", "amplitude", "detuning", "register"}



def add_observables(sequence: pulser.Sequence, t_list: list[int], observables: list[str],
                    num_of_samples: int, emulator: str) -> dict:

    obs_list = {}
    for obs in observables:
        match obs.lower():
            case 'state':
                #First create the sequence
                max_sequence = sequence.get_duration()

                #Then order it
                evaluation_times = [t/max_sequence for t in t_list]

                #Create the state
                state = None

                match emulator.lower():
                    case 'pulser':
                        state = pulser.backend.StateResult(evaluation_times=evaluation_times)
                    case 'sv':
                        state = emu_sv.StateResult(evaluation_times=evaluation_times)
                    case 'mps':
                        state = emu_mps.StateResult(evaluation_times=evaluation_times)
                    case _:
                        raise Exception(f"Emulator {emulator} is not available. Please choose one of the following: ['pulser', 'sv', 'mps']")

                #Add state to dictionary
                obs_list['state'] = state

            case 'bitstring':
                #First create the sequence
                max_sequence = sequence.get_duration()

                #Then order it
                evaluation_times = [t/max_sequence for t in t_list]

                #Create the bitstring
                bitstrings = None
                match emulator.lower():
                    case 'pulser':
                        bitstrings = pulser.backend.BitStrings(evaluation_times=evaluation_times, num_shots=num_of_samples)
                    case 'sv':
                        bitstrings = emu_sv.BitStrings(evaluation_times=evaluation_times, num_shots=num_of_samples)
                    case 'mps':
                        bitstrings = emu_mps.BitStrings(evaluation_times=evaluation_times, num_shots=num_of_samples)
                    case _:
                        raise Exception(f"Emulator {emulator} is not available. Please choose one of the following: ['pulser', 'sv', 'mps']")

                #Add bitstring to dictionary
                obs_list['bitstring'] = bitstrings
            case _:
                raise Exception("So far, nothing else is supported")


    return obs_list

##################################
###### Noise model settings ######
##################################
def get_noise_model(noise_model_params) -> NoiseModel:

    noise_model = None

    if noise_model_params is None:
        noise_model = NoiseModel(dephasing_rate=0.008*2*np.pi, relaxation_rate=0.0025*2*np.pi)
        #noise_model = NoiseModel(p_false_neg=0.05,p_false_pos=0.05)

    else:
        noise_model = pulser.NoiseModel.from_abstract_repr(noise_model_params)

    return noise_model


##################################
##### Initial state settings #####
##################################

#Initialise the state
def create_emulator_initial_state_dict(node_features: list[int], num_qubits: int):

    initial_state_val_list = ['g' for _ in range(num_qubits)]
    if node_features:
        initial_state_val_list = ['g' if feature == 1 else 'r' for feature in node_features]

    initial_state_val = ''.join(initial_state_val_list)
    return initial_state_val

##################################
###### Setting up emulators ######
##################################


#CPU emulator provided by pulser
def pulser_setup(sequence: pulser.Sequence, t_list: list[int], node_features: list[int], num_of_samples: int):


    #Create the observables list
    obs_list = add_observables(sequence, t_list, observables=['state','bitstring'], num_of_samples=num_of_samples, emulator='pulser')

    #Create the config
    config = pulser.backend.EmulationConfig(observables=list(obs_list.values()))

    #Create the backend
    backend = pulser.backends.QutipBackendV2(sequence, config=config)

    #Initialise the state (NOTE: For pulser, you need to initialise the backend first, in order to initialise the state)
    if node_features:

        #Set up the state
        init_state = qutip.tensor([qutip.basis(2, node_feature) for node_feature in node_features])
        backend._sim_obj.set_initial_state(init_state)


    return backend

#CPU emulator provided by pulser
def pulser_setup_v2(sequence: pulser.Sequence, t_list: list[int], node_features: list[int], num_of_samples: int, allow_noise: bool, noise_model_params):


    init_state = qutip.tensor([qutip.basis(2, 1) for _ in range(len(sequence.qubit_info))])
    if node_features:
        init_state = qutip.tensor([qutip.basis(2, node_feature) for node_feature in node_features])


    #Create the observables list
    obs_list = add_observables(sequence, t_list, observables=['state', 'bitstring'], num_of_samples=num_of_samples, emulator='pulser')

    config = None
    if allow_noise:
        noise_model = get_noise_model(noise_model_params)

        #Create the config including noise
        config = pulser.backend.EmulationConfig(observables=list(obs_list.values()), noise_model=noise_model)

    else:

        #Create a noiseless config
        config = pulser.backend.EmulationConfig(observables=list(obs_list.values()))

    #Create the backend
    backend = pulser_simulation.QutipBackendV2(sequence, config=config)
    backend._sim_obj.set_initial_state(init_state)

    return backend, obs_list['bitstring']


#State vector GPU accelerator
def sv_setup(sequence: pulser.Sequence, t_list: list[int], node_features: list[int],
             num_of_samples: int,
             allow_noise: bool, noise_model_params,
             dt: int = 5):

    #Initialise the state (NOTE: For pulser, you need to initialise the backend first, in order to initialise the state)
    n = len(sequence.qubit_info)
    initial_state_config = create_emulator_initial_state_dict(node_features,n)

    amplitude = 1.0
    init_state = emu_sv.StateVector.from_state_amplitudes(eigenstates=('r','g'), amplitudes={initial_state_config:amplitude} )

    #Obtain the observables list
    observables_dict = add_observables(sequence, t_list, observables=['state','bitstring'], num_of_samples=num_of_samples, emulator='sv')

    config = None
    if allow_noise:
        noise_model = get_noise_model(noise_model_params)

        for type_of_noise in noise_model.noise_types:
            if type_of_noise not in NON_LINDBLADIAN_NOISE:

                #Change the initial state to a matrix, if we have non-lindbladian noise.
                init_state = emu_sv.DensityMatrix.from_state_vector(init_state)
                break

        #Create the configuration file with noise
        config = emu_sv.SVConfig(dt=dt, observables=list(observables_dict.values()), log_level=2000, initial_state=init_state, noise_model=noise_model)
    else:

        #Create a noiseless configuration file
        config = emu_sv.SVConfig(dt=dt, observables=list(observables_dict.values()), log_level=2000, initial_state=init_state)

    #Create the backend
    backend = emu_sv.SVBackend(sequence, config=config)
    return backend, observables_dict['bitstring']

#Matrix Product State emulator
def mps_setup(sequence: pulser.Sequence, t_list: list[int], node_features: list[int],
              num_of_samples: int,
              allow_noise: bool, noise_model_params,
              dt: int = 25):


    #Initialise the state (NOTE: For pulser, you need to initialise the backend first, in order to initialise the state)
    n = len(sequence.qubit_info)
    initial_state_config = create_emulator_initial_state_dict(node_features,n)

    amplitude = 1.0
    init_state = emu_mps.MPS.from_state_amplitudes(eigenstates=('r','g'), amplitudes={initial_state_config: amplitude})

    #Obtain the observables list
    observables_dict = add_observables(sequence, t_list, observables=['bitstring'], num_of_samples=num_of_samples, emulator='mps')

    #Create the configuration file
    mpsconfig = None
    if allow_noise:
        noise_model = get_noise_model(noise_model_params)

        #Create the config including noise
        mpsconfig = emu_mps.MPSConfig(
            dt = dt,
            observables=list(observables_dict.values()),
            noise_model=noise_model,
            initial_state=init_state,
            optimize_qubit_ordering=False,
            log_level=logging.INFO
        )
    else:
        mpsconfig = emu_mps.MPSConfig(
            dt = dt,
            observables=list(observables_dict.values()),
            initial_state=init_state,
            optimize_qubit_ordering=False,
            log_level=logging.INFO,
            precision=1e-5,
            solver=Solver.TDVP,
            interaction_cutoff=1e-6,
            #max_bond_dim=20
            )

    backend = emu_mps.MPSBackend(sequence, config=mpsconfig)

    return backend, observables_dict['bitstring']



##################################
## Transform the distributions ###
##################################

def graph_recenter(graph_coordinates):

    #Get the boundries of the graph coordinates
    min_coords = np.min(graph_coordinates[:,0:2], axis=0)
    max_coords = np.max(graph_coordinates[:,0:2], axis=0)

    #Find the new center by using the mean of the boundies
    new_center_point = np.mean(np.vstack((max_coords, min_coords)), axis=0)

    graph_coordinates[:,0:2] -= new_center_point
    return graph_coordinates

#######################################################
#### Subsampling on events that are close in space ####
#######################################################

def remove_pixels_given_criteria(graph_coordinates: np.ndarray[np.ndarray],
                                 initial_features: dict[str,int],
                                 event_to_row: dict[str,int],
                                 events_to_delete_info_dict: dict[str, int]):

    #Create a dictionary of all the events.
    #If value = -1, then this event is not deleted.
    #If value >= 0, then the events is deleted by the remained_event, which is specified by 'value'.
    deleted_by_dict = {key: -1 for key in events_to_delete_info_dict.keys()}

    events_to_delete = []

    for initial_event, possible_removals_list in events_to_delete_info_dict.items():

        if initial_event not in events_to_delete:

            for event_to_be_removed in possible_removals_list:

                #Start the process of deleting the event, unless it is already deleted
                if event_to_be_removed not in events_to_delete:

                    deleted_by_dict[event_to_be_removed] = int(initial_event) #Store the remained_event, that caused the deletion of the current one

                    graph_coordinates[event_to_row[event_to_be_removed]] = np.inf #Omit the entires in the coordinates matrix
                    del initial_features[event_to_be_removed] #Delete the entry from the features list

                    events_to_delete.append(event_to_be_removed) #Add this to the list that stores deleted events

    new_graph_coordinates = graph_coordinates[~np.all(graph_coordinates == np.inf, axis=1)]

    #for event in events_to_delete:
    #    events_to_delete_info_dict[event] #Note: this does not seem to do anything useful

    return new_graph_coordinates, initial_features, deleted_by_dict


def subsample_spike_close_coordinate(graph_coordinates: np.ndarray[np.ndarray],
                                     initial_features: dict[str,int],
                                     threshold_radius: float) -> tuple[np.ndarray, dict[str,int], dict[tuple[float,float], list[str]]]:

    #First create the dictionary to row mapping (it is one to one)
    event_to_row = {}
    row_to_event = {}

    for row, key in enumerate(initial_features.keys()):
        event_to_row[key] = row
        row_to_event[row] = key

    #Calculate the distance between the points
    distance_between_points = cdist(graph_coordinates, graph_coordinates)
    n = graph_coordinates.shape[0]
    all_close_events = {}

    for i in range(n):
        close_events_list = []

        for j in range(i+1, n): #Look at the top right matrix
            if distance_between_points[i,j] < threshold_radius:
                close_events_list.append(row_to_event[j])

        all_close_events[row_to_event[i]] = close_events_list

    return remove_pixels_given_criteria(graph_coordinates,initial_features,event_to_row,all_close_events)



def subsample_far_coordinates(graph_coordinates: np.ndarray[np.ndarray],
                              initial_features: dict[str,int],
                              allowed_radial_distance: float):



    #Find the nodes that break the condition
    are_events_out_of_region = np.linalg.norm(graph_coordinates, axis=1) > allowed_radial_distance
    pre_deleted_values = {key: -2 for key in initial_features.keys()}

    if np.any(are_events_out_of_region): #If any event is out of the region

        for event_order, event in enumerate(initial_features.keys()): #Iterate through all the events

            if are_events_out_of_region[event_order]: #Event is out of the specified region.

                warnings.warn("Event: " + event + f" is out of the radial region {allowed_radial_distance} µm. Its length is {np.linalg.norm(graph_coordinates, axis=1)[event_order]}")

                #Save the value to a separate list
                pre_deleted_values[event] = initial_features[event]

                #Mark the element in the graph as infinity, so it can be deleted at the end
                graph_coordinates[event_order] = np.inf

        #Delete the event
        for event, val in pre_deleted_values.items():
            if val >= -1 and val <= 1:
                del initial_features[event]

        new_graph_coordinates = graph_coordinates[~np.all(graph_coordinates == np.inf, axis=1)] #Delete the new events

    else:

        #Send back the features without altering them
        new_graph_coordinates = graph_coordinates

    return new_graph_coordinates, initial_features, pre_deleted_values



def expand_far_coordinates(initial_features: dict[str,int],
                           pre_deleted_values: dict[str,int]):


    events_to_return = {}
    for events, is_event_deleted in pre_deleted_values.items():

        if is_event_deleted >= -1 and is_event_deleted <= 1:
            events_to_return[events] = is_event_deleted
        else:
            events_to_return[events] = initial_features[events]

    return events_to_return


#Getting the feature value, based on the event that caused this event to be deleted
def expand_spike_per_pixel(initial_features: dict[str,int], deleted_events: dict[str,int]):


    #Create a new list of features that will be returned at the end
    returned_feautres = {}

    for existing_event, deleted_by in deleted_events.items():

        if deleted_by >= 0: #This means the event was deleted by the specified event
            returned_feautres[existing_event] = initial_features[str(deleted_by)]
        else:
            returned_feautres[existing_event] = initial_features[existing_event]

    return returned_feautres



#######################################################
##### Get information about the graph coordinates #####
#######################################################
def obtain_graph_coordinates(graph: dict[int, tuple[float, float]] | dict[int, tuple[float, float, float, float]], initial_features: dict[str,int],
                             recenter: bool, dimensions: int,
                             mult_coeff: float, linear_multiplicative_time_c: float):


    graph_coordinates = np.array(list(graph.values()))


    if dimensions == 2:
        time_frames = graph_coordinates[:,2].reshape(-1,1)
        graph_coordinates = graph_coordinates[:,:2]
        graph_coordinates *= mult_coeff
        if linear_multiplicative_time_c != 0:
            graph_coordinates += linear_multiplicative_time_c*time_frames
    else:
        graph_coordinates[:,0:2] *= mult_coeff #NOTE: time is disregarded in this case:

    if recenter:
        graph_coordinates = graph_recenter(graph_coordinates)

    return graph_coordinates

### Construct the dynamics of each graph.
def sample_single_run_num_excited_qubits(result_outcomes: collections.Counter[str,int], sample_num: int, graph_size: int) -> np.ndarray:

    qubit_num_excited_elems = np.zeros(graph_size + 1)
    for bin_key, num_outcome in result_outcomes.items():
        index = bin_key.count('1') # Count the number of ones
        qubit_num_excited_elems[index] += num_outcome


    qubit_num_excited_elems /= sample_num
    return qubit_num_excited_elems

#Define a method using constant time, constant detuning and constant amplitude
def sample_single_distribution_v1(graph: dict[tuple[float,float]], graph_num_vertex,
                               dev,
                               t: int = 660, rabi_freq: float = 2*np.pi, detuning_freq: float = 2*np.pi*0.7, phase: float = 0.0,
                               num_of_samples = 1000, draw_graph: bool = False):

    graph_coordinates = list(graph.values())
    #Create the register
    reg = pulser.Register.from_coordinates(
        graph_coordinates,
        prefix="q",  # All qubit IDs will start with 'q'
        center=True,
    )

    if draw_graph:
        reg.draw(blockade_radius=dev.rydberg_blockade_radius(1),
                    draw_graph = True,
                    draw_half_radius=True)
    #Create the sequence
    seq = pulser.Sequence(reg, dev)

    #Declare the channels that are needed
    seq.declare_channel("rydberg_global","rydberg_global")

    #Create and add the pulse
    constant_pulse = pulser.Pulse.ConstantPulse(
        t,
        rabi_freq,
        detuning_freq,
        phase
    )
    seq.add(constant_pulse, "rydberg_global")


    #Execute the system
    backend = pulser.backends.QutipBackend(seq)
    result = backend.run()

    #Compute the runs from the system
    result_distribution = result.sample_state(t/1000, num_of_samples)

    #Compute the \sum_i n_i
    P_distribution = sample_single_run_num_excited_qubits(result_distribution, num_of_samples, graph_num_vertex)

    return P_distribution


def sample_single_distribution_v2(graph: dict[tuple[float,float]], graph_num_vertex,
                                  dev,
                                  t_list: list[int] = [50,50,50,50,50], rabi_freq: float = 2*np.pi, detuning_freq: float = 2*np.pi*0.7, phase: float = 0.0,
                                  num_of_samples = 1000, draw_graph: bool = False):

    graph_coordinates = list(graph.values())

    #Create the register
    reg = pulser.Register.from_coordinates(
        graph_coordinates,
        prefix="q",  # All qubit IDs will start with 'q'
        center=True,
    )

    if draw_graph:
        reg.draw(blockade_radius=dev.rydberg_blockade_radius(1),
                    draw_graph = True,
                    draw_half_radius=True)

    #Create the sequence
    seq = pulser.Sequence(reg, dev)

    #Declare the channels that are needed
    seq.declare_channel("rydberg_global","rydberg_global")

    for i, t in enumerate(t_list):


        amplitude_param = rabi_freq
        det_param = detuning_freq
        phase_param = phase

        if (i % 2 == 1):
            amplitude_param = 0
            det_param = 0
            phase_param = 0

        #Create and add the pulse
        constant_pulse = pulser.Pulse.ConstantPulse(
            t,
            amplitude_param,
            det_param,
            phase_param
        )


        seq.add(constant_pulse, "rydberg_global")


    #Execute the system
    backend = pulser.backends.QutipBackend(seq)
    result = backend.run()

    #Compute the runs from the system
    result_distribution = result.sample_state(t/1000, num_of_samples)

    #Compute the \sum_i n_i
    P_distribution = sample_single_run_num_excited_qubits(result_distribution, num_of_samples, graph_num_vertex)

    return P_distribution


#Define a method using constant time, constant detuning and constant amplitude

def sample_single_distribution_v3(graph: dict[int, tuple[float, float]] | dict[int, tuple[float, float, float, float]], graph_num_vertex: int,
                                  dev: pulser.devices.Device,
                                  initial_features: dict[str,int] = [],
                                  t_list: list[int] = [200, 400, 600, 800], rabi_freq: float = 2*np.pi, detuning_freq: float = 2*np.pi*0.7, phase: float = 0.0,
                                  num_of_samples = 100, draw_graph: bool = False,
                                  recenter: bool = True, dimensions: int = 2,
                                  mult_coeff: float = 1.0, linear_multiplicative_time_c: float = 0,
                                  remove_close_points: bool = False,
                                  allow_noise: bool = False, noise_model_params = None,
                                  detuning_encoding: bool = False) -> list[np.ndarray] | tuple[list[np.ndarray],list[dict[str,int]]]:


    ##### Set up the graph coordinates
    graph_coordinates = obtain_graph_coordinates(graph, initial_features,
                                                 recenter, dimensions,
                                                 mult_coeff, linear_multiplicative_time_c)


    deleted_events : dict  = None

    if remove_close_points:
        reduce_factor = dev.min_atom_distance
        if dev.min_atom_distance == 0:
            reduce_factor = dev.rydberg_blockade_radius(rabi_freq)/np.sqrt(2)

        graph_coordinates, initial_features, deleted_events = subsample_spike_close_coordinate(graph_coordinates, initial_features, reduce_factor)


    deleted_radial_events = None
    if dev.max_radial_distance > 0:
        graph_coordinates, initial_features, deleted_radial_events = subsample_far_coordinates(graph_coordinates, initial_features, dev.max_radial_distance)


    #Remove points that are outside of the region
    graph_size = len(graph_coordinates)
    print(f"Number of qubits: {graph_size}")

    #If we use the emulator, check what type of emulator we need
    emu_sv = None
    gpu_acceleration = None

    if graph_size <= 9:
        gpu_acceleration = False
    elif (graph_size <= 24 and (not allow_noise)) or (allow_noise and graph_size <= 12):
        gpu_acceleration = True #Emulate using SV simulator
        emu_sv = True
    else:
        gpu_acceleration = True #Emulate using MPS
        emu_sv = False


    labels_to_use = None
    node_features = None

    #Set up the initial features, if it is needed
    if initial_features: #Only when we have initial features, we should extract the labels. In that case, we preserve the old behaviour

        #Initialise the node features
        labels_to_use = []
        node_features = []
        for node_label, feature in initial_features.items():
            labels_to_use.append(node_label)
            node_features.append(feature)


    t_max = max(t_list)
    #Create the register
    reg = None
    if dimensions == 2:
        reg = pulser.Register.from_coordinates(
            graph_coordinates,
            labels=labels_to_use,
            center=(not recenter),
        )
    else:
        reg = pulser.Register3D.from_coordinates(
            graph_coordinates,
            labels=labels_to_use,
            center=(not recenter),
        )

    if draw_graph:
        reg.draw(blockade_radius=dev.rydberg_blockade_radius(1),
                    draw_graph = True,
                    draw_half_radius=True)

    #Create the sequence
    seq = pulser.Sequence(reg, dev)

    #Declare the channels that are needed
    seq.declare_channel("rydberg_global","rydberg_global")

    #Create and add the pulse
    print(f"t_max: {t_max}")
    constant_pulse = pulser.Pulse.ConstantPulse(
        t_max,
        rabi_freq,
        detuning_freq,
        phase
    )
    seq.add(constant_pulse, "rydberg_global")


    if detuning_encoding:

        #First, get the map from the register, where the qubits are mapped
        detuning_map = reg.define_detuning_map(
            {labels_to_use[i] : (1 - node_features[i]) for i in range(len(labels_to_use))}
        ) #Map is defined as: No detuning if qubit = 1, and detuning with weight if qubit = 0 => Mimicking the reversal of normal encoding.

        #Second, apply the map to the sequence
        seq.config_detuning_map(detuning_map, "dmm_0")

        #Thirdly, define the weight
        local_detuning_weight = 0.1*rabi_freq #According to the "Practical Quantum Reservoir Computing in Rydberg Atom Arrays" paper

        #Then, we apply the pulse to the channel
        seq.add_dmm_detuning(pulser.ConstantWaveform(t_max, -local_detuning_weight), "dmm_0")

        #Finally, remove the features, so that the encoding at initialisation does not occur.
        node_features = None

    bitstrings = None

    #Specify the backend
    backend = None

    if gpu_acceleration:
        if emu_sv:
            backend, bitstrings = sv_setup(seq, t_list, node_features, num_of_samples, allow_noise=allow_noise, noise_model_params=noise_model_params)
        else:
            backend, bitstrings = mps_setup(seq, t_list, node_features, num_of_samples, allow_noise=allow_noise, noise_model_params=noise_model_params)
    else:
        backend, bitstrings = pulser_setup_v2(seq, t_list, node_features, num_of_samples, allow_noise=allow_noise, noise_model_params=noise_model_params)

    start_test_time = time.time()

    result = backend.run()

    end_test_time = time.time()

    print(f"It takes {end_test_time - start_test_time} seconds for running the simulator for {graph_size} qubits")

    #Compute the runs from the system
    start_test_time = time.time()
    p_dists = []
    most_likely_states = []
    for i, t in enumerate(t_list):

        #Initialise the distribution
        result_distribution = None
        if gpu_acceleration:
            result_distribution = result.get_result(bitstrings,t/t_max)
        else:
            time_to_check = result.get_result_times(bitstrings)[i]
            result_distribution = result.get_result(bitstrings,time_to_check)
            #current_state = result.state[i]
            #result_distribution = current_state.sample(num_shots=num_of_samples)

        #Compute the \sum_i n_i
        p_distribution = sample_single_run_num_excited_qubits(result_distribution, num_of_samples, graph_num_vertex)

        if initial_features:

            #Obtain the most likely outcome
            bitstr = result_distribution.most_common(1)[0][0]
            features_at_time_t = {}
            for i, node_label in enumerate(labels_to_use):
                features_at_time_t[node_label] = 1 - int(bitstr[i])


            ###Expand events that are outside the region
            if not (deleted_radial_events is None):
                features_at_time_t = expand_far_coordinates(features_at_time_t, deleted_radial_events)


            ###Add the events that are moved by neighboring events
            if remove_close_points:
                if not initial_features:
                    raise Exception("Shouldn't try and remove close points if you have not set the initial features")
                features_at_time_t = expand_spike_per_pixel(features_at_time_t, deleted_events)

            most_likely_states.append(features_at_time_t)

        p_dists.append(p_distribution)





    end_test_time = time.time()

    print(f"It takes {end_test_time - start_test_time} seconds for getting the result for {graph_size} qubits")


    if initial_features:
        return p_dists, most_likely_states
    return p_dists




#Method to compute the existing graphs
#Return: An array containing the dynamics of the distribution



def create_graph_distributions(graph_labels_list: list[dict[str,tuple]], dev,
                               t: Union[int, list[int]], rabi_freq: float = 2*np.pi, detuning_freq: float = 2*np.pi*0.7, phase: float = 0.0,
                               num_of_samples = 10000):


    graph_p_distributions = []

    #Find the maximum graph element
    max_graph_elements = max([len(graph) for graph, _ in graph_labels_list])
    for graph in graph_labels_list:


        dg = False

        if isinstance(t, list):
            single_p_dist = sample_single_distribution_v2(graph[0], max_graph_elements, dev, t, rabi_freq, detuning_freq, phase, num_of_samples, draw_graph=dg)
        else:
            single_p_dist = sample_single_distribution_v1(graph[0], max_graph_elements, dev, t, rabi_freq, detuning_freq, phase, num_of_samples, draw_graph=dg)

        graph_p_distributions.append((single_p_dist, graph[1]))

    return graph_p_distributions



def create_graph_distributions_v2(graph_labels_list: list[ tuple[ dict[int, tuple[float, float, float, float]], int]], dev,
                                  t: list[int], rabi_freq: float = 2*np.pi, detuning_freq: float = 2*np.pi*0.7, phase: float = 0.0,
                                  num_of_samples = 10000) -> list[tuple[list[np.ndarray], int]]:


    graph_p_distributions = []

    #Find the maximum graph element
    max_graph_elements = max([len(graph) for graph, _ in graph_labels_list])
    for graph, label in graph_labels_list:

        dg = False

        single_p_dists = sample_single_distribution_v3(graph, max_graph_elements, dev, t, rabi_freq, detuning_freq, phase, num_of_samples, draw_graph=dg)

        graph_p_distributions.append((single_p_dists, label))

    return graph_p_distributions

##################################
###### Construct the kernel ######
##################################



#### Define a run of constructing a single Kernel
def run_kernel(graphs_dataset, device, kern_func, eval_func, t: Union[int, list[int]], rabi_freq: float, train_test_cutoff: float = 0.8):

    graph_p_dist = create_graph_distributions(graphs_dataset, device, t=t,rabi_freq=rabi_freq)

    classifier = svm.SVC(kernel=kern_func)#, C=5)


    cutoff = int(train_test_cutoff*len(graph_p_dist))

    dist_labels_train = graph_p_dist[:cutoff]
    dist_labels_test =  graph_p_dist[cutoff:]


    train_dist = [dist for dist, _ in dist_labels_train]
    train_labels = [label for _, label in dist_labels_train]


    test_dist = [dist for dist, _ in dist_labels_test]
    test_labels = [label for _, label in dist_labels_test]

    classifier.fit(train_dist, train_labels)


    score_result = eval_func(classifier.predict(train_dist),np.array(train_labels))
    return classifier, score_result, (train_dist, train_labels), (test_dist, test_labels)


def find_optimal_time_evolution(train_dists: list[list[np.ndarray]], train_labels: list[int], possible_times: list[int],
                                kern_func, eval_func, save_evol_time: str) -> tuple[svm.SVC, float, tuple[list[np.ndarray], list[int]], int, int]:


    #Get all the possible times
    possible_times_len = len(possible_times)


    optimal_t = 0
    optimal_score = 0
    optimal_t_index = -1

    classifiers_list = []


    score_q_evol_list = []
    for i in range(possible_times_len):

        #Create the distribution dataset, based on the time
        train_distribution = [distrib[i] for distrib in train_dists]

        #Create the classifier and fit the data
        classifier = svm.SVC(kernel=kern_func)
        classifier.fit(train_distribution, train_labels)

        #Find the score
        score_result = eval_func(classifier.predict(train_distribution), np.array(train_labels))

        score_q_evol_list.append(score_result)

        classifiers_list.append(classifier)

        #Change the approprate time
        if optimal_score < score_result:

            optimal_t = possible_times[i]
            optimal_score = score_result
            optimal_t_index = i

    train_dist = [distrib[optimal_t_index] for distrib in train_dists]

    opt_classifier = classifiers_list[optimal_t_index]

    if save_evol_time:
        score_q_evol = [list(possible_times), list(score_q_evol_list)]

        with open(save_evol_time, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(score_q_evol)

    return opt_classifier, optimal_score, train_dist, optimal_t, optimal_t_index


#### Define a run of constructing a single Kernel
def run_kernel_v2(graphs_dataset: list[ tuple[ dict[int, tuple[float, float, float, float]], int]], device,
                  kern_func, eval_func,
                  t: tuple[int], rabi_freq: float, train_test_cutoff: float = 0.8) -> tuple[svm.SVC, float, tuple, tuple, int]:


    #Obtain the possible times
    possible_times = list(range(t[1], t[0], -t[2]))


    #Create the graphs distributions: Nx(MxP,L), where
    #N = Different graphs
    #M = Distribution list, based on different time
    #P = The distribution
    #L = Label
    graph_p_dists = create_graph_distributions_v2(graphs_dataset, device, t=possible_times,rabi_freq=rabi_freq)


    #Create the cutoff value
    cutoff = int(train_test_cutoff*len(graph_p_dists))

    #Seperate the test and the training data
    dist_labels_train = graph_p_dists[:cutoff]
    dist_labels_test =  graph_p_dists[cutoff:]

    #Separate the training distributions and the labels
    train_dists = [dists for dists, _ in dist_labels_train]
    train_labels = [label for _, label in dist_labels_train]

    #Separate the testing distributions and the labels
    test_dists = [dists for dists, _ in dist_labels_test]
    test_labels = [label for _, label in dist_labels_test]


    classifier, optimal_score, train_dist, optimal_t, optimal_t_index = find_optimal_time_evolution(train_dists=train_dists,train_labels=train_labels,possible_times=possible_times,
                                                                                                          kern_func=kern_func,eval_func=eval_func)

    test_dist = [distrib[optimal_t_index] for distrib in test_dists]

    return classifier, optimal_score, (train_dist, train_labels), (test_dist, test_labels), optimal_t



#####################################################################################
################################## Main Program #####################################
#####################################################################################


def QFM_single_time_run(graphs_dataset,
                        kernel_function, eval_function,
                        min_t: int, max_t: int, delta_t: int):

    dev = pulser.AnalogDevice

    max_score = 0
    optimal_t = 0
    for t in range(max_t, min_t, -delta_t):

        _, f1_score_metric, _, _ = run_kernel(graphs_dataset, dev, kernel_function, eval_function, t, 4*np.pi)


        if f1_score_metric >= max_score:
            max_score = f1_score_metric
            optimal_t = t


    ### Evaluate the model on testing dataset
    classifier, score, train_data, test_data = run_kernel(graphs_dataset, dev, kernel_function, eval_function, optimal_t, 4*np.pi)

    return classifier, score, train_data, test_data, optimal_t


def QFM_single_time_run_v2(graphs_dataset: list[ tuple[ dict[int, tuple[float, float, float, float]], int]],
                           kernel_function, eval_function,
                           min_t: int, max_t: int, delta_t: int):

    dev = pulser.AnalogDevice

    ### Evaluate the model on testing dataset
    classifier, score, train_data, test_data, optimal_t = run_kernel_v2(graphs_dataset, dev, kernel_function, eval_function, (min_t, max_t, delta_t), 4*np.pi)

    return classifier, score, train_data, test_data, optimal_t