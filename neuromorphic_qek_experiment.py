import numpy as np
import random
import csv
import argparse
import json

from model import evalutaion_funcs as ef
from model import neuromorphic_quantum_kernel as nqkrn
from model import kernel as krn
from model import synthetic_data_creation as sdc

def parse_args():

    #Collect the parsing arguments
    parser = argparse.ArgumentParser(description="Train and evalute QA-AEGNN on synthetic dataset")

    ### Synthetic Dataset Parameters
    parser.add_argument("--dataset_size", type=int, default=200, help="The number of graphs in the dataset")
    parser.add_argument("--graph_size", type=int, default=100, help="The number of nodes in the graph")
    parser.add_argument("--w_weight", type=float, default=0.1, help="The weight to deviate from the graph set")

    ### Model parameters
    parser.add_argument("--window_size", type=int, default=10, help="The window frame size")
    parser.add_argument("--min_q_time", type=int, default=50, help="Minimum quantum time evolution")
    parser.add_argument("--max_q_time", type=int, default=2000, help="Maximum quantum time evolution")
    parser.add_argument("--q_delta_t", type=int, default=50, help="Quantum time evolution difference")
    parser.add_argument("--quantum_device", type=str, default='default', help="Type of quantum device")
    parser.add_argument("--feature_encoding", type=str, default='init_state', help="Type of feature encoding")
    
    ### Number of executions
    parser.add_argument("--num_runs", type=int, default=1, help="The number of different experiment executions")

    ### Include noise
    parser.add_argument("--noise_type", type=str, default='None', help="Specify the noise type")

    ### Misc
    parser.add_argument("--csvnum", type=int, help="Gets the csv file number")
    parser.add_argument("--seed", type=int, default=881, help="Set the seed number")
    parser.add_argument("--save_q_str", type=bool, default=False, help="Record and save accuracy against time evolution")

    return parser.parse_args()



def main():

    #Obtaim the arguments
    args = parse_args()

    #Seed
    seed_number = args.seed
    random.seed(seed_number)

    offset = 0
    if args.csvnum:
        offset = args.csvnum

    #Define the parameters for the dataset
    num_nodes = args.graph_size
    num_dataset = args.dataset_size
    p0 = args.w_weight

    #Define GNN parameters
    delta_T = args.window_size
    slide_size = delta_T/2

    #Quantum evolution parameters
    min_t = args.min_q_time
    max_t = args.max_q_time
    q_evolution_delta_t = args.q_delta_t
    quantum_device_type = args.quantum_device

    timesteps = np.arange(0,num_nodes,slide_size).tolist()

    #Specify type of feature encoding
    feature_encoding = args.feature_encoding
    
    detuning_encoding = None
    match feature_encoding.lower():
        case 'init_state':
            detuning_encoding = False
        case 'detuning':
            detuning_encoding = True
        case _:
            raise Exception("Other type of feature encoding are not supported.")

    #Specify a noise model
    noise_type = args.noise_type

    noise_model = None
    allow_noise = None
    match noise_type.lower():
        case 'spam':
            allow_noise = True
        case 'quantum':
            allow_noise = True
            with open("noise_models/noise_model.json", "r") as json_file:
                noise_model = json.load(json_file)
        case _:
            allow_noise = False


    #Specify the kernel and the score function
    error_func = ef.standard_error_rate
    kernel_function = krn.construct_kernel


    num_runs = args.num_runs

    test_scores = []

    for i in range(num_runs):

        test_scores_per_num_nodes = []

        #Create the graphs
        if seed_number == 0: #No randomness
            graphs_dataset = sdc.create_dataset(num_nodes, num_dataset, p0, allow_red_start=False, starting_point=(0,0,0,1), seed=seed_number,allow_repeat=False)
        else:
            graphs_dataset = sdc.create_dataset(num_nodes, num_dataset, p0, allow_red_start=False, starting_point=(0,0,0,1), seed=(seed_number+num_runs*(offset)+i), allow_repeat=False) # Create different runs with the seed number

        q_time_save_str = ''
        if args.save_q_str:
            q_time_save_str = 'results/q_time_evolution_data/graph_data_save_' + str(num_runs*offset + i + 1)

        #Call the neural network
        classifiers, train_score_results, (state_vec_train, _), (state_vec_test, labels_test), _ = nqkrn.execute_kernel_run_v2(graphs_dataset, time_steps=timesteps, delta_t=delta_T,
                                                                                                                                                               kernel_function=kernel_function, error_func=error_func,
                                                                                                                                                               q_evolution_min_t=min_t,q_evolution_max_t=max_t,q_evolution_delta_t=q_evolution_delta_t,quantum_device_type=quantum_device_type,
                                                                                                                                                               legacy_features_init_type='last_layer',
                                                                                                                                                               save_q_evol_time=q_time_save_str, detuning_encoding=detuning_encoding,
                                                                                                                                                               noise_model_params=noise_model, allow_noise=allow_noise)

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

            test_scores_per_num_nodes.append(test_score_results)


        print(f"Final score on training set for run {i}: {train_score_results}")
        print(f"Final score on test set for run {i}: {test_scores_per_num_nodes[-1]}")

        test_scores.append(test_scores_per_num_nodes)


    #Save the testing data from the majority rule
    run_number = str(offset)
    data_to_save = []
    main_info = ['Test RUN', *timesteps]
    data_to_save.append(main_info)


    for i in range(num_runs):
        test_scores_per_num_nodes = test_scores[i]
        row_info = [i+1, *test_scores_per_num_nodes]
        data_to_save.append(row_info)

    with open('results/neuromorphic_qaaegnn_' + run_number + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_to_save)


    #Save the testing data for each window
    data_to_save = []
    main_info = ['Test RUN', *timesteps]
    data_to_save.append(main_info)


if __name__ == "__main__":
    main()