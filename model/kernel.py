import numpy as np
from sklearn import svm

from . import tools
from . import qaaegnn


##################################
######## Helper Functions ########
##################################

def majority_rule_predict(vec_to_predict: list, classifier_vec: list, num_windows: int) -> list[float]:


    test_vote_array = np.array([])

    for i in range(num_windows):

        input_test_dataset_per_window = [input_test_data[i] for input_test_data in vec_to_predict]

        predicted_test_labels = classifier_vec[i].predict(input_test_dataset_per_window)

        if i == 0:
            test_vote_array = predicted_test_labels
        else:
            test_vote_array = np.vstack((test_vote_array, predicted_test_labels))

    final_prediction = []
    if num_windows == 1:
        final_prediction = test_vote_array.tolist()
    else:
        final_prediction = [tools.count_vote(test_vote_array[:,i]) for i in range(len(vec_to_predict))]

    return final_prediction


def single_window_predict(vec_to_predict: list[np.ndarray], classifier: svm.SVC) -> list[float]:

    predicted_test_labels = classifier.predict(vec_to_predict)
    return predicted_test_labels.tolist()

def majority_rule_predict_reuse(test_predictions_list: np.ndarray):

    final_prediction = []
    if len(test_predictions_list.shape) == 1:
        final_prediction = test_predictions_list.tolist()
    else:
        final_prediction = [tools.count_vote(test_predictions_list[:,i]) for i in range(test_predictions_list.shape[1])]

    return final_prediction


##################################
######## Kernel Function #########
##################################


#### Create the different measurements

def shannon_entropy(P):
    return -sum([p*np.log(p) if p > 0 else 0 for p in P])

def jensen_shannon_div(P1, P2):


    if len(P1) != len(P2):
        raise Exception("The lengths of the distributions should be the same")


    #Create the individual shannon entropies
    shannon_P1 = shannon_entropy(P1)
    shannon_P2 = shannon_entropy(P2)

    #Create the combined shannon entropy
    combined_P = [(P1[i] + P2[i])/2 for i in range(len(P1))]
    shannon_P12 = shannon_entropy(combined_P)

    #Return the overall entropy

    return shannon_P12 - (shannon_P1 + shannon_P2)/2

#Create the Kernel from the equation provided
def construct_kernel(x_mat1, x_mat2):

    m = len(x_mat1)
    n = len(x_mat2)
    kernel = np.empty((m,n))

    for i in range(m):
        for j in range(n):
            kernel[i,j] = np.exp(-jensen_shannon_div(x_mat1[i],x_mat2[j]))
    return kernel

##################################
######### Kernel (Main) ##########
##################################


def run_kernel(input_dataset, labels, kernel_func, eval_func, train_test_cutoff: float = 0.8):

    classifier = svm.SVC(kernel=kernel_func, max_iter=10000)

    cutoff = int(train_test_cutoff*len(input_dataset))

    input_train_dataset = input_dataset[:cutoff]
    input_test_dataset  = input_dataset[cutoff:]

    labels_train = labels[:cutoff]
    labels_test  = labels[cutoff:]

    classifier.fit(input_train_dataset, labels_train)


    score_results = eval_func(classifier.predict(input_train_dataset),np.array(labels_train))

    return classifier, score_results, (input_train_dataset, labels_train), (input_test_dataset, labels_test)


#Return a probability vector, outlining the probability of the class
def run_kernel_majority_rule(input_dataset, labels, kernel_func, eval_func, num_windows: int, train_test_cutoff: float = 0.8):


    cutoff = int(train_test_cutoff*len(input_dataset))

    input_train_dataset = input_dataset[:cutoff]
    input_test_dataset  = input_dataset[cutoff:]

    labels_train = labels[:cutoff]
    labels_test  = labels[cutoff:]

    vote_array = np.array([])

    classifier_vec = []
    for i in range(num_windows):

        input_train_dataset_per_window = [input_train_data[i] for input_train_data in input_train_dataset]

        classifier = svm.SVC(kernel=kernel_func, max_iter=10000,C=1)

        classifier.fit(input_train_dataset_per_window, labels_train)

        classifier_vec.append(classifier)
        predicted_labels_train = classifier.predict(input_train_dataset_per_window)

        if i == 0:
            vote_array = predicted_labels_train
        else:
            vote_array = np.vstack((vote_array, predicted_labels_train))


    final_prediction = [tools.count_vote(vote_array[:,i]) for i in range(len(input_train_dataset))]

    score_results = eval_func(final_prediction,np.array(labels_train))

    return classifier_vec, score_results, (input_train_dataset, labels_train), (input_test_dataset, labels_test), final_prediction


#Return a probability vector, outlining the probability of the class
def run_kernel_val_set_majority_rule(input_dataset, labels, kernel_func, eval_func, num_windows: int, train_test_cutoff: float = 0.8, train_validation_cutoff: float = 0.8):


    #Testing cutoff
    test_cutoff = int(train_test_cutoff*len(input_dataset))

    input_train_val_dataset = input_dataset[:test_cutoff]
    input_test_dataset  = input_dataset[test_cutoff:]

    labels_train_val = labels[:test_cutoff]
    labels_test  = labels[test_cutoff:]

    #Validation cutoff
    validation_cutoff = int(train_validation_cutoff*len(input_train_val_dataset))

    input_train_dataset = input_train_val_dataset[:validation_cutoff]
    input_validation_dataset = input_train_val_dataset[validation_cutoff:]

    labels_train = labels_train_val[:validation_cutoff]
    labels_validation = labels_train_val[validation_cutoff:]


    vote_array_train = np.array([])
    vote_array_val = np.array([])
    vote_array_test = np.array([])

    classifier_vec = []
    for i in range(num_windows):

        input_train_dataset_per_window = [input_train_data[i] for input_train_data in input_train_dataset]

        classifier = svm.SVC(kernel=kernel_func, max_iter=10000,C=1)

        classifier.fit(input_train_dataset_per_window, labels_train)

        classifier_vec.append(classifier)
        predicted_labels_train = classifier.predict(input_train_dataset_per_window)


        #Get the validation cost
        input_val_dataset_per_window = [input_val_data[i] for input_val_data in input_validation_dataset]
        predicted_labels_val = classifier.predict(input_val_dataset_per_window)

        #Get the testing cost
        input_test_dataset_per_window = [input_test_data[i] for input_test_data in input_test_dataset]
        predicted_labels_test = classifier.predict(input_test_dataset_per_window)

        #Record the data
        if i == 0:
            vote_array_train = predicted_labels_train
            vote_array_val = predicted_labels_val
            vote_array_test = predicted_labels_test
        else:
            vote_array_train = np.vstack((vote_array_train, predicted_labels_train))
            vote_array_val = np.vstack((vote_array_val, predicted_labels_val))
            vote_array_test = np.vstack((vote_array_test, predicted_labels_test))


    #Get the final counting prediction
    final_prediction_train = [tools.count_vote(vote_array_train[:,i]) for i in range(len(input_train_dataset))]
    final_prediction_val = [tools.count_vote(vote_array_val[:,i]) for i in range(len(input_validation_dataset))]
    final_prediction_test = [tools.count_vote(vote_array_test[:,i]) for i in range(len(input_test_dataset))]

    score_results_train = eval_func(final_prediction_train,np.array(labels_train))
    score_results_val = eval_func(final_prediction_val,np.array(labels_validation))
    score_results_test = eval_func(final_prediction_test,np.array(labels_test))

    return classifier_vec, (score_results_train, score_results_val, score_results_test), (input_train_dataset, labels_train), (input_validation_dataset, labels_validation), (input_test_dataset, labels_test), final_prediction_train, final_prediction_val


def find_optimal_alpha(train_dists: list[list[np.ndarray]], train_labels: list[int], possible_alphas: list[float],
                       kern_func, eval_func) -> tuple[svm.SVC, float, tuple[list[np.ndarray], list[int]], int, int]:


    #Get all the possible times
    possible_times_len = len(possible_alphas)

    optimal_alpha = 0
    optimal_score = 0
    optimal_alpha_index = -1

    classifiers_list = []

    alpha_score_list = []
    for i in range(possible_times_len):

        #Create the distribution dataset, based on the time
        train_distribution = [distrib[i] for distrib in train_dists]

        #Create the classifier and fit the data
        classifier = svm.SVC(kernel=kern_func)
        classifier.fit(train_distribution, train_labels)

        #Find the score
        score_result = eval_func(classifier.predict(train_distribution), np.array(train_labels))

        alpha_score_list.append(score_result)

        classifiers_list.append(classifier)

        #Change the approprate time
        if optimal_score < score_result:

            optimal_alpha = possible_alphas[i]
            optimal_score = score_result
            optimal_alpha_index = i

    train_dist = [distrib[optimal_alpha_index] for distrib in train_dists]

    opt_classifier = classifiers_list[optimal_alpha_index]

    return opt_classifier, optimal_score, train_dist, optimal_alpha, optimal_alpha_index


def run_kernel_optimal_alpha(input_dataset, labels,
                             kernel_func, eval_func,
                             alpha_vecs: list[float], train_test_ratio: float = 0.8,):

    #Create the training and the testing dataset
    cutoff = int(train_test_ratio*len(input_dataset))

    input_train_dataset = input_dataset[:cutoff]
    input_test_dataset  = input_dataset[cutoff:]

    labels_train = labels[:cutoff]
    labels_test  = labels[cutoff:]


    #Use quantum evolution kernel method to return the optimal classifier
    classifier, score_result_train, optimal_train_distribution_per_window, optimal_alpha, alpha_index = find_optimal_alpha(input_train_dataset, labels_train,
                                                                                                                           alpha_vecs,
                                                                                                                           kernel_func, eval_func)


    #Predict the testing data, using the current prediction
    optimal_test_distribution_per_window = [distrib[alpha_index] for distrib in input_test_dataset]

    predicted_labels_train = classifier.predict(optimal_train_distribution_per_window)


    return classifier, score_result_train, (optimal_train_distribution_per_window, labels_train), (optimal_test_distribution_per_window, labels_test), predicted_labels_train, optimal_alpha, alpha_index


#Return a probability vector, outlining the probability of the class
def run_kernel_quantum(input_dataset, labels,
                       kernel_func, eval_func,
                       quantum_time_evolution: list[int], train_test_ratio: float = 0.8, save_q_evol_time: str = '') -> tuple[svm.SVC, float, tuple[list[list[float]], list[int]], tuple[list[list[float]], list[int]], list[int], int]:

    #Create the training and the testing dataset
    cutoff = int(train_test_ratio*len(input_dataset))

    input_train_dataset = input_dataset[:cutoff]
    input_test_dataset  = input_dataset[cutoff:]

    labels_train = labels[:cutoff]
    labels_test  = labels[cutoff:]


    #Use quantum evolution kernel method to return the optimal classifier
    classifier, score_result_train, optimal_train_distribution_per_window, qevolution_time, qevol_index = qaaegnn.find_optimal_time_evolution(input_train_dataset, labels_train,
                                                                                                                                          quantum_time_evolution,
                                                                                                                                          kernel_func, eval_func, save_q_evol_time)

    #Predict the testing data, using the current prediction
    optimal_test_distribution_per_window = [distrib[qevol_index] for distrib in input_test_dataset]

    predicted_labels_train = classifier.predict(optimal_train_distribution_per_window)

    return classifier, score_result_train, (optimal_train_distribution_per_window, labels_train), (optimal_test_distribution_per_window, labels_test), predicted_labels_train, qevolution_time, qevol_index

#Return a probability vector, outlining the probability of the class, based on the majority rule
def run_kernel_majority_rule_quantum(input_dataset, labels, kernel_func, eval_func,
                                     num_windows: int, quantum_time_evolution: list[int],
                                     train_test_cutoff: float = 0.8, save_q_evol_time: str = '') -> tuple[list[svm.SVC], float, tuple[list[list[float]], list[int]], tuple[list[list[float]], list[int]], list[int], list[int]]:


    cutoff = int(train_test_cutoff*len(input_dataset))

    input_train_dataset = input_dataset[:cutoff]
    input_test_dataset  = input_dataset[cutoff:]

    labels_train = labels[:cutoff]
    labels_test  = labels[cutoff:]

    vote_array = np.array([])

    classifier_vec = []

    qevolution_optimal_time_list = []

    optimal_input_features_train = []
    optimal_input_features_test = []

    for i in range(num_windows):


        #Obtain the training and the testing dataset
        input_train_dataset_per_window = [input_train_data[i] for input_train_data in input_train_dataset]
        input_test_dataset_per_window = [input_test_data[i] for input_test_data in input_test_dataset]


        #Use quantum evolution kernel method to return the optimal classifier
        classifier, _, optimal_train_distribution_per_window, qevolution_time, qevol_index = qaaegnn.find_optimal_time_evolution(input_train_dataset_per_window, labels_train,
                                                                                                                                         quantum_time_evolution,
                                                                                                                                         kernel_func, eval_func, save_q_evol_time)

        optimal_test_distribution_per_window = [distrib[qevol_index] for distrib in input_test_dataset_per_window]


        #Predict the training set, in order to create the vote list
        predicted_labels_train = classifier.predict(optimal_train_distribution_per_window)


        #Save the optimal time
        classifier_vec.append(classifier)

        qevolution_optimal_time_list.append(qevolution_time)

        optimal_input_features_train.append(optimal_train_distribution_per_window)
        optimal_input_features_test.append(optimal_test_distribution_per_window)

        if i == 0:
            vote_array = predicted_labels_train
        else:
            vote_array = np.vstack((vote_array, predicted_labels_train))


    final_prediction = [tools.count_vote(vote_array[:,i]) for i in range(len(input_train_dataset))]

    score_results = eval_func(final_prediction,np.array(labels_train))

    return classifier_vec, score_results, (optimal_input_features_train, labels_train), (optimal_input_features_test, labels_test), final_prediction, qevolution_optimal_time_list
