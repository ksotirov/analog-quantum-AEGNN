import numpy as np
import time
import random
import csv
import argparse
import json

from model import evalutaion_funcs as ef
from model import neuromorphic_quantum_kernel as nqkrn
from model import kernel as krn
from model import synthetic_data_creation as sdc



#Collect the parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--csvnum", type=int, help="Gets the csv file number")

args = parser.parse_args()

#Seed
#seed_number = 0 #No randomness
#seed_number = 42
#seed_number = 799
seed_number = 881
random.seed(seed_number)

start = time.time()


offset = 0
if args.csvnum:
    offset = args.csvnum

#Define the parameters for the dataset
num_nodes = 100
num_per_set = 100
p0 = 0.1
dist = 5 #NOTE: This parameter is changed

#Define GNN parameters
delta_T = 10
slide_diff = 2
slide_size = delta_T/slide_diff
num_layers = 4

#Quantum evolution parameters
device_type = 'default'
min_t = 50
max_t = 2001
q_evolution_delta_t = 50

timesteps = np.arange(0,num_nodes,slide_size).tolist()

#Specify a noise model
noise_model = None
noise_model_params = False
if noise_model_params:
    with open("noise_models/noise_model.json", "r") as json_file:
        noise_model = json.load(json_file)


#Specify the kernel and the score function
error_func = ef.standard_error_rate
kernel_function = krn.construct_kernel


num_runs = 2 #Test with 50

train_scores = []
test_scores = []
test_scores_independent = []


for i in range(num_runs):

    train_scores_per_num_nodes = []
    test_scores_per_num_nodes = []
    test_scores_per_window = []

    #Create the graphs
    if seed_number == 0:
        graphs_dataset = sdc.create_dataset(num_nodes, num_per_set, p0, dist, allow_red_start=False, starting_point=(0,0,0,1), seed=seed_number,allow_repeat=False)
    else:
        graphs_dataset = sdc.create_dataset(num_nodes, num_per_set, p0, dist, allow_red_start=False, starting_point=(0,0,0,1), seed=(seed_number+5*(offset)+i), allow_repeat=False)

    q_time_save_str = 'q_time_evolution_data/graph_data_save_' + str(2*offset + i + 1)

    #Call the neural network
    classifiers, train_score_results, (state_vec_train, labels_train), (state_vec_test, labels_test), quantum_evolution_time = nqkrn.execute_kernel_run_v2(graphs_dataset, time_steps=timesteps, delta_t=delta_T,
                                                                                                                                                           kernel_function=kernel_function, error_func=error_func,
                                                                                                                                                           q_evolution_min_t=min_t,q_evolution_max_t=max_t,q_evolution_delta_t=q_evolution_delta_t,device_type=device_type,legacy_features_init_type='last_layer',
                                                                                                                                                           save_q_evol_time=q_time_save_str, detuning_encoding=False,
                                                                                                                                                           noise_model_params=noise_model, allow_noise=False)

    test_predictions_list = np.array([])
    for shift_windows in range(len(classifiers)):

        print(timesteps[:shift_windows + 1])

        current_classifier = classifiers[shift_windows]
        current_state_vec_train = state_vec_train[shift_windows]
        current_state_vec_test = state_vec_test[shift_windows]

        #Predict the test vector
        test_prediction_per_window = krn.single_window_predict(current_state_vec_test, current_classifier)

        if shift_windows == 0:
            test_predictions_list = np.array(test_prediction_per_window)
        else:
            test_predictions_list = np.vstack((test_predictions_list, test_prediction_per_window))


        test_prediction_final = krn.majority_rule_predict_reuse(test_predictions_list)


        test_score_results = error_func(test_prediction_final, np.array(labels_test))
        test_score_results_per_window = error_func(test_prediction_per_window, np.array(labels_test))



        test_scores_per_num_nodes.append(test_score_results)
        test_scores_per_window.append(test_score_results_per_window)


        print(f"Optimal time for this window: {quantum_evolution_time[shift_windows]}")
        print(f"Final score on the training set: {train_score_results}")
        print(f"Score on the trainig set for current window: {error_func(krn.single_window_predict(current_state_vec_train, current_classifier),np.array(labels_train))} ")
        print(f"Score on the testing set using Majority rule: {test_score_results}")
        print(f"Score on the testing set for current window: {test_score_results_per_window}")

    test_scores.append(test_scores_per_num_nodes)
    test_scores_independent.append(test_scores_per_window)



#Save the testing data from the majority rule
run_number = str(10 + offset)
data_to_save = []
main_info = ['Test RUN', *timesteps]
data_to_save.append(main_info)


for i in range(num_runs):
    test_scores_per_num_nodes = test_scores[i]
    row_info = [i+1, *test_scores_per_num_nodes]
    data_to_save.append(row_info)

with open('qaaegnn_neuromorphic/neuromorphic_qaaegnn_test_majority_rule_' + run_number + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_save)


#Save the testing data for each window
data_to_save = []
main_info = ['Test RUN', *timesteps]
data_to_save.append(main_info)


for i in range(num_runs):
    test_scores_per_window = test_scores_independent[i]
    row_info = [i+1, *test_scores_per_window]
    data_to_save.append(row_info)

with open('qaaegnn_neuromorphic/neuromorphic_qaaegnn_test_independent_' + run_number + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_save)

end = time.time()
print(f"The time of the program is: {(end - start)} seconds.")