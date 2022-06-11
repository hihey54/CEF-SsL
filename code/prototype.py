import os
import time
import math
from pathlib import Path
from pprint import pprint
import time
import sys
cwd = os.getcwd()
from functions import *
import sklearn
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


F_holdout = 0.20   # FUTURE SET. Percentage of samples (taken from each individual set, malicious AND benign) to put in F.
n, k = 2, 3 # number of runs of CEF-SSL
Cm_list = [1, 2, 5]   # COST OF LABELLING. 1=malicious and benign samples have same cost; 2=malicious samples have twice the cost of benign samples; 5=malicious samples have five times the cose of benign samples
Lm_list = [10, 20] # LABELLING BUDGET (in malicious samples). The script will choose the provided number of malicious samples, and then pick the benign samples according to the Cm value
active_budget = 'regular' # either 'regular' or a custom integer
activeLearning_trials = 2 # how many times each "active learning" model is retrained on new randomly chosen samples (falling in the confidence thresholds)
al_confidences = ['high', 'mid', 'low']
high_confidence_window = 0.2 # samples whose confidence is ABOVE (100-high_confidence_window/2) are considered to be of "high confidence"
low_confidence_window = 0.2 # samples whose confidence is BELOW (low_confidence_window/2) are considered to be of "high confidence"
# choose the basic classification algorithm here. 
## IMPORTANT: to make it work with the implemented SsL techniques, the classifier must support the "predict_proba" method!
base_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                 min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-2, 
                 random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
# base_clf = HistGradientBoostingClassifier() ### a potential alternative

###### LOAD DATASET

features = pd.Index(['feature1', 'feature2', '...']) # list of features
label_name = 'Result' # name of the "label" column
dataset_file = "path.to.dataset.file" #path of dataset file
df = load_dataset(dataset_file) # load dataset into a dataframe. NOTE: the "load_dataset" method is just an abstraction and must be provided by the user
malicious_families = ["family1"] # list of malicious families

# benign_df = load_dataset(benign_dataset_file)
# malicious_files = ["maliciious_dataset_file1", "malicious_dataset_file2"]
# malicious_df_list = []
# for index, family in enumerate(malicious_families):
#    malicious_df_list.append(load_dataset(malicious_files[index]))


### NOTE: the EVALUATION requires the following two additional inputs: "benign_df" and "malicious_df". Ideally, the latter should be those belonging
### to a specific attack family (CEF-SsL is tailored for binary classification: a sample is either benign or malicious)

### These inputs must be manually determined by the user: they can be provided by loading files that contain only benign/malicious samples; 
### or they can be provided by "extracting" from the df the samples corresponding to a specific label
benign_df = df[df[label_name]==False] 
malicious_df = df[df[label_name]==True]


########## BEGIN EVALUATION ##########

# the script is executed for each malicious family (in the case a dataset contains multiple "attacks"; otherwise, it will be run just once)
for index, family in enumerate(malicious_families):
    begin = time.time()
    print("Benign:{}\tMalicious:{}\t({})".format(len(benign_df), len(malicious_df), family))
    for Cm in Cm_list:
        for Lm in Lm_list:
            for reiterate in range(k):
                # We create a new F
                print("reiterate: {} on {}".format(reiterate, k))
                malicious_df['holdout'] = (np.random.uniform(0,1,len(malicious_df)) <= F_holdout)
                benign_df['holdout'] = (np.random.uniform(0,1,len(benign_df)) <= F_holdout)
                malicious_F_df, malicious_UL_df = malicious_df[malicious_df['holdout']==True], malicious_df[malicious_df['holdout']==False]
                benign_F_df, benign_UL_df = benign_df[benign_df['holdout']==True], benign_df[benign_df['holdout']==False]
                
                
                F_df = pd.concat([malicious_F_df, benign_F_df])
                
                print("Size of F (test data): {} ben {} mal ({} tot)".format(len(benign_F_df), len(malicious_F_df), len(F_df)))
                
                F_labels = F_df[label_name]
                UL_df = pd.concat([malicious_UL_df, benign_UL_df])
                UL_labels = UL_df[label_name]
                for run in range(n):
                    print("\trun {} of {} (for reiterate: {} on {})".format(run, n, reiterate, k))
                    Lb = int(Lm * Cm)
                    Lcost = Lb + Lm
                    malicious_L_df = malicious_UL_df.sample(n=Lm)
                    benign_L_df = benign_UL_df.sample(n=Lb)
                    malicious_U_df = malicious_UL_df.drop(malicious_L_df.index)
                    benign_U_df = benign_UL_df.drop(benign_L_df.index)
                    L_df = pd.concat([malicious_L_df, benign_L_df])
                    L_labels = L_df[label_name]
                    U_df = pd.concat([malicious_U_df, benign_U_df])
                    U_labels = U_df[label_name]
                    
                    print("\t\tSL (UpperBound). Large L: {}b {}m\t"
                          .format(len(benign_L_df)+len(benign_U_df), len(malicious_L_df)+len(malicious_U_df), len(UL_df)), end=" ")
                    SL_clf, SL_precision, SL_recall, SL_fscore, SL_trainTime, SL_testTime, SL_Fpredictions = train_test(train_df=UL_df, train_labels=UL_labels, 
                                                                                         test_df=F_df, test_labels=F_labels, features=features, messages=1, base_clf=base_clf)
                    print("F1: {:.5f} TrainTime: {:.3f}s".format(SL_fscore, SL_trainTime))
                    
                    print("\t\tsl (LowerBound). L: {}b {}m\t"
                          .format(len(benign_L_df), len(malicious_L_df), len(L_df)), end="")
                    sl_clf, sl_precision, sl_recall, sl_fscore, sl_trainTime, sl_testTime, sl_Fpredictions = train_test(train_df=L_df, train_labels=L_labels, 
                                                                                         test_df=F_df, test_labels=F_labels, features=features, messages=1, base_clf=base_clf)
                    print("F1: {:.5f} TrainTime: {:.3f}s".format(sl_fscore, sl_trainTime))
                    #########################
                    ##### VANILLA Pseudo Labelling
                    ## PREPARE    
                    start_time = time.time()
                    sl_Upredictions, sl_Uprobabilities = sl_clf.predict(U_df[features]), sl_clf.predict_proba(U_df[features])
                    sl_predictTime = time.time() - start_time

                    # Create column for Predictions on dataframes
                    U_df['sl_prediction'] = sl_Upredictions
                    L_df['sl_prediction'] = L_df[label_name] # dummy column (the pseudo-labelling models will use this column as 'label')

                    # Assign predicted probability (higher probability=more likely that it is benign)
                    U_df['sl_probability'] = (np.hsplit(sl_Uprobabilities,2))[0]

                    # Assign confidence based on (predicted) probability and split dataframe
                    U_df, U_high_df, U_mid_df, U_low_df = assign_confidence(U_df, high_confidence_window, low_confidence_window, 
                                                                                        probability_column_name = 'sl_probability', confidence_column_name = 'sl_confidence', debug=False, split=True)
            

                    pseudoAll_L_df = pd.concat([U_df, L_df])
                    pseudoAll_L_labels = pseudoAll_L_df['sl_prediction']
                    
                    
                    ## TRAIN and TEST
                    print("\t\tBaseline ssl (all pseudo labels). L: {}b {}m\t"
                          .format(len(benign_L_df), len(malicious_L_df)), end="")
                    ssl_clf, ssl_precision, ssl_recall, ssl_fscore, ssl_trainTime, ssl_testTime, ssl_Fpredictions = train_test(train_df=pseudoAll_L_df, train_labels=pseudoAll_L_labels, 
                                                                                         test_df=F_df, test_labels=F_labels, features=features, messages=1, base_clf=base_clf)
                    ssl_trainTime = ssl_trainTime + sl_predictTime + sl_trainTime
                    print("F1: {:.5f} TrainTime: {:.3f}s".format(ssl_fscore, ssl_trainTime))
                    
                    pseudoHigh_L_df = pd.concat([U_high_df, L_df])
                    pseudoHigh_L_labels = pseudoHigh_L_df['sl_prediction']
                    print("\t\tpssl (high confidence pseudo labels). L: {}b {}m\t".
                          format(len(benign_L_df), len(malicious_L_df)), end="")
                    pssl_clf, pssl_precision, pssl_recall, pssl_fscore, pssl_trainTime, pssl_testTime, pssl_Fpredictions = train_test(train_df=pseudoHigh_L_df, train_labels=pseudoHigh_L_labels, 
                                                                                         test_df=F_df, test_labels=F_labels, features=features, messages=1, base_clf=base_clf)
                    pssl_trainTime = pssl_trainTime + sl_predictTime + sl_trainTime
                    print("F1: {:.5f} TrainTime: {:.3f}s".format(pssl_fscore, pssl_trainTime))
                    
                    U_midlow_df = pd.concat([U_mid_df, U_low_df])
                    start_time = time.time()
                    ssl_Umlpredictions, ssl_Umlprobabilities = ssl_clf.predict(U_midlow_df[features]), ssl_clf.predict_proba(U_midlow_df[features])
                    ssl_predictTime = time.time() - start_time
                    U_midlow_df['ssl_prediction'] = ssl_Umlpredictions
                    pseudoHigh_L_df['ssl_prediction'] = pseudoHigh_L_df['sl_prediction'] # dummy column (the retrained pseudo-labelling models will use this column as 'label')
                    U_midlow_df['ssl_probability'] = (np.hsplit(ssl_Umlprobabilities,2))[0]
                    U_midlow_df, U_midlow_high_df, U_midlow_mid_df, U_midlow_low_df = assign_confidence(U_midlow_df, high_confidence_window, low_confidence_window, 
                                                                                        probability_column_name = 'ssl_probability', confidence_column_name = 'ssl_confidence', debug=False, split=True)
                    pseudoHigh_high_L_df = pd.concat([pseudoHigh_L_df, U_midlow_high_df])
                    pseudoHigh_high_L_labels = pseudoHigh_high_L_df['ssl_prediction']
                    
                    
                    print("\t\trpssl (high confidence pseudo labels, twice). L: {}b {}m\t".
                          format(len(benign_L_df), len(malicious_L_df)), end="")
                    rpssl_clf, rpssl_precision, rpssl_recall, rpssl_fscore, rpssl_trainTime, rpssl_testTime, rpssl_Fpredictions = train_test(train_df=pseudoHigh_high_L_df, train_labels=pseudoHigh_high_L_labels, 
                                                                                         test_df=F_df, test_labels=F_labels, features=features, messages=1, base_clf=base_clf)
                    rpssl_trainTime = rpssl_trainTime + pssl_trainTime
                    print("F1: {:.5f} TrainTime: {:.3f}s".format(rpssl_fscore, rpssl_trainTime))
                    
                    ############ ACTIVE LEARNING
                    #### This is for the support model to use for active learning
                    # generating support L
                    malicious_support_L_df = malicious_L_df.sample(n=int(Lm * active_budget[0]))
                    benign_support_L_df = benign_L_df.sample(n=int(Lb * active_budget[1]))
                    support_L_df = pd.concat([malicious_support_L_df, benign_support_L_df])
                    support_L_labels = support_L_df[label_name]
                    
                    # regenerating U
                    malicious_UL_df = malicious_df[malicious_df['holdout']==False]
                    benign_UL_df = benign_df[benign_df['holdout']==False]
                    malicious_U_df = malicious_UL_df.drop(malicious_support_L_df.index)
                    benign_U_df = benign_UL_df.drop(benign_support_L_df.index)
                    support_U_df = pd.concat([malicious_U_df, benign_U_df])
                    support_U_labels = support_U_df[label_name]
                    
                    # training support sl and predicting labels on the (new) U
                    support_sl_clf, support_sl_precision, support_sl_recall, support_sl_fscore, support_sl_trainTime, support_sl_testTime, support_sl_Fpredictions = train_test(train_df=support_L_df, train_labels=support_L_labels, 
                                                                                         test_df=F_df, test_labels=F_labels, features=features, messages=0, base_clf=base_clf)
                    
                    start_time = time.time()
                    support_sl_Upredictions, support_sl_Uprobabilities = support_sl_clf.predict(support_U_df[features]), support_sl_clf.predict_proba(support_U_df[features])
                    support_sl_predictTime = time.time() - start_time
                    support_U_df['support_sl_prediction'] = support_sl_Upredictions
                    support_L_df['support_sl_prediction'] = support_L_df[label_name] # dummy column (the pseudo-labelling models will use this column as 'label')
                    support_U_df['support_sl_probability'] = (np.hsplit(support_sl_Uprobabilities,2))[0]

                    # Assign confidence based on (predicted) probability and split dataframe
                    support_U_df, support_U_high_df, support_U_mid_df, support_U_low_df = assign_confidence(support_U_df, high_confidence_window, low_confidence_window, 
                                                                                        probability_column_name = 'support_sl_probability', confidence_column_name = 'support_sl_confidence', debug=False, split=True)
                    
                    
                    # Train and Test - VANILLA Active Learning
                    leftout_budget = int(Lm * (1-active_budget[0])) + int(Lb * (1-active_budget[1]))
                    
                    if 'high' in al_confidences:
                        print("\t\tahssl (active-high confidence)... L: {}b {}m + {} leftout...\t".
                          format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget), end=" ")
                        ahssl_precision, ahssl_recall, ahssl_fscore, ahssl_trainTime = active_learning(confidences_df=support_U_high_df, 
                                                                                                          baseTrain_df=support_L_df, 
                                                                                                          validation_df=F_df, 
                                                                                                          features=features, base_clf=base_clf,
                                                                                                          trials=activeLearning_trials, ssl=False, samples=leftout_budget, label_name = label_name, messages=1)
                        ahssl_trainTime = ahssl_trainTime + support_sl_predictTime + support_sl_trainTime
                        print("F1: {:.5f} TrainTime: {:.3f}s".format(ahssl_fscore, ahssl_trainTime))
                        
                    if 'mid' in al_confidences:
                        print("\t\taossl (active-mid confidence)... L: {}b {}m + {} leftout...\t".
                          format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget), end=" ")
                        aossl_precision, aossl_recall, aossl_fscore, aossl_trainTime = active_learning(confidences_df=support_U_mid_df, 
                                                                                                           baseTrain_df=support_L_df, 
                                                                                                           validation_df=F_df, 
                                                                                                           features=features, base_clf=base_clf,
                                                                                                           trials=activeLearning_trials, ssl=False, samples=leftout_budget, label_name = label_name, messages=1)
                        aossl_trainTime = aossl_trainTime + support_sl_predictTime + support_sl_trainTime
                        print("F1: {:.5f} TrainTime: {:.3f}s".format(aossl_fscore, aossl_trainTime))
                    if 'low' in al_confidences:
                        print("\t\talssl (active-low confidence)... L: {}b {}m + {} leftout...\t".
                          format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget), end=" ")
                        alssl_precision, alssl_recall, alssl_fscore, alssl_trainTime = active_learning(confidences_df=support_U_low_df, 
                                                                                                           baseTrain_df=support_L_df, 
                                                                                                           validation_df=F_df, 
                                                                                                           features=features, base_clf=base_clf,
                                                                                                           trials=activeLearning_trials, ssl=False, samples=leftout_budget, label_name = label_name, messages=1)
                        alssl_trainTime = alssl_trainTime + support_sl_predictTime + support_sl_trainTime
                        print("F1: {:.5f} TrainTime: {:.3f}s".format(alssl_fscore, alssl_trainTime))
                    ############
                    #### PREPARE: Pseudo Labelling + Active Learning
                    
                    ## First, train the support pseudo-labelling model using the labels with high-confidence
                    support_L_df['support_sl_prediction'] = support_L_df[label_name]
                    pseudoHigh_support_L_df = pd.concat([support_U_high_df, support_L_df])
                    pseudoHigh_support_L_labels = pseudoHigh_support_L_df['support_sl_prediction']
                    
                    
                    print("\t\tPseudo-Active Learning: the base size of the training set is: {}".format(len(pseudoHigh_support_L_df)))
                    support_pssl_clf, support_pssl_precision, support_pssl_recall, support_pssl_fscore, support_pssl_trainTime, support_pssl_testTime, support_pssl_Fpredictions = train_test(train_df=pseudoHigh_support_L_df, train_labels=pseudoHigh_support_L_labels, 
                                                                                         test_df=F_df, test_labels=F_labels, features=features, messages=0, base_clf=base_clf)
                    support_pssl_trainTime = support_pssl_trainTime + support_sl_trainTime + support_sl_predictTime
                    
                    ## Then, use such support pseudo-labelling to predict the confidence of the remaining samples in U
                    support_U_midlow_df = pd.concat([support_U_mid_df, support_U_low_df])
                    start_time = time.time()
                    support_pssl_Umlpredictions, support_pssl_Umlprobabilities = support_pssl_clf.predict(support_U_midlow_df[features]), support_pssl_clf.predict_proba(support_U_midlow_df[features])
                    support_pssl_predictTime = time.time() - start_time
                    support_U_midlow_df['support_ssl_prediction'] = support_pssl_Umlpredictions
                    pseudoHigh_support_L_df['support_ssl_prediction'] = pseudoHigh_support_L_df['support_sl_prediction'] # dummy column (the retrained pseudo-labelling models will use this column as 'label')
                    support_U_midlow_df['support_ssl_probability'] = (np.hsplit(support_pssl_Umlprobabilities,2))[0]
                    support_U_midlow_df, support_U_midlow_high_df, support_U_midlow_mid_df, support_U_midlow_low_df = assign_confidence(support_U_midlow_df, high_confidence_window, low_confidence_window,
                                                                                                                                        probability_column_name = 'support_ssl_probability', confidence_column_name = 'support_ssl_confidence', debug=False, split=True)
                    
                    ## TRAIN and TEST - Pesudo-Active Learning
                    if 'high' in al_confidences:
                        print("\t\tpahssl (pseudoActive-high confidence)... L: {}b {}m + {} leftout...\t".
                          format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget), end=" ")
                        pahssl_precision, pahssl_recall, pahssl_fscore, pahssl_trainTime = active_learning(confidences_df=support_U_midlow_high_df, 
                                                                                                          baseTrain_df=pseudoHigh_support_L_df, 
                                                                                                          validation_df=F_df, 
                                                                                                          features=features, base_clf=base_clf,
                                                                                                          trials=activeLearning_trials, ssl=True, samples=leftout_budget, label_name = label_name, messages=1)
                        pahssl_trainTime = pahssl_trainTime + support_pssl_trainTime + support_pssl_predictTime
                        print("F1: {:.5f} TrainTime: {:.3f}s".format(pahssl_fscore, pahssl_trainTime))
                    if 'mid' in al_confidences:
                        print("\t\tpahssl (pseudoActive-mid confidence)... L: {}b {}m + {} leftout...\t".
                          format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget), end=" ")
                        paossl_precision, paossl_recall, paossl_fscore, paossl_trainTime = active_learning(confidences_df=support_U_midlow_mid_df, 
                                                                                                           baseTrain_df=pseudoHigh_support_L_df, 
                                                                                                           validation_df=F_df, 
                                                                                                           features=features, base_clf=base_clf,
                                                                                                           trials=activeLearning_trials, ssl=True, samples=leftout_budget, label_name = label_name, messages=1)
                        paossl_trainTime = paossl_trainTime + support_pssl_trainTime + support_pssl_predictTime
                        print("F1: {:.5f} TrainTime: {:.3f}s".format(paossl_fscore, paossl_trainTime))
                    if 'low' in al_confidences:
                        print("\t\tpahssl (pseudoActive-low confidence)... L: {}b {}m + {} leftout...\t".
                          format(len(benign_support_L_df), len(malicious_support_L_df), leftout_budget), end=" ")
                        palssl_precision, palssl_recall, palssl_fscore, palssl_trainTime = active_learning(confidences_df=support_U_midlow_low_df, 
                                                                                                           baseTrain_df=pseudoHigh_support_L_df, 
                                                                                                           validation_df=F_df, 
                                                                                                           features=features, base_clf=base_clf,
                                                                                                           trials=activeLearning_trials, ssl=True, samples=leftout_budget, label_name = label_name, messages=1)
                        palssl_trainTime = palssl_trainTime + support_pssl_trainTime + support_pssl_predictTime
                        print("F1: {:.5f} TrainTime: {:.3f}s".format(palssl_fscore, palssl_trainTime))
runtime = time.time() - begin
print("Runtime total: {:.5f}".format(runtime))


########## END EVALUATION ##########