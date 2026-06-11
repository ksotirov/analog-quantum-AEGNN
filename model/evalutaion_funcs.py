import numpy as np


##################################
####### Evaluate the score #######
##################################

### Define the F1 Score
def f1_score(predicted_labels, true_labels):


    len_true = len(true_labels)
    if len(predicted_labels) != len_true:
        raise Exception("The length of the two labels must be the same")


    true_positive = wrong_predictions = 0
    for i in range(len_true):
        if (predicted_labels[i] == 1) and (true_labels[i] == 1):
            true_positive += 1

        elif predicted_labels[i]*true_labels[i] == -1: #Prediction and true labels is wring
            wrong_predictions += 1

    return true_positive/(true_positive + (wrong_predictions/2))


### Define the standard error rate
def standard_error_rate(predicted_labels: np.ndarray | list, true_labels: np.ndarray | list):


    ##### Exception checking (Begin) #####
    #Convert the data types from list to np.arrays
    if isinstance(predicted_labels, list):
        predicted_labels = np.array(predicted_labels)

    if isinstance(true_labels, list):
        true_labels = np.array(true_labels)

    #Check if both of them are 1 dimensional arrays
    if predicted_labels.ndim != 1:
        raise Exception(f"The predicted labels are expected to be 1 dimensional array. Instead, it was an {predicted_labels.ndim} dimensional array")

    if true_labels.ndim != 1:
        raise Exception(f"The true labels are expected to be 1 dimensional array. Instead, it was an {predicted_labels.ndim} dimensional array")
    

    len_true = true_labels.shape[0]

    #Check if the two vectors are the same length
    if predicted_labels.shape[0] != len_true:
        raise Exception("The length of the two labels must be the same")


    #Check if all the elements are -1 or 1
    if ((predicted_labels == 1) | (predicted_labels == -1)).sum() != predicted_labels.shape[0]:
        raise Exception("The predicted labels vector contains entries that are not -1 or 1")
    
    if ((true_labels == 1) | (true_labels == -1)).sum() != true_labels.shape[0]:
        raise Exception("The true labels vector contains entries that are not -1 or 1")
    ###### Exception checking (End) ######


    return np.mean(np.multiply(predicted_labels, true_labels) == 1)
