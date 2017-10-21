
from Dataset import *
from Model import Model
import os, inspect
from numpy import load
import pandas as pd
from sklearn.tree.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor
from numpy import *
import time
from operator import itemgetter
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from numpy.random import *
import re
import os, pwd
import numpy as np
import gc
import time
import argparse as argp
import timeit



def mse_err(a, b):
    assert(a.shape == b.shape)
    #return np.sqrt(sum(np.square(a - b)))
    return (sum(np.square(a - b)))


# if os.getlogin()=="root":
#     plt.switch_backend('agg')
#or os.getlogin = lambda: pwd.getpwuid(os.getuid())[0] https://stackoverflow.com/questions/4399617/python-os-getlogin-problem
#plt.switch_backend('agg')

def main(ml_model, *args):
    datasetname = args[4]
    timehorizon = args[3]
    test_size = args[5]
    prior_file = args[6]
    prior_type = args[7]
    datatype = args[8]
    bias_score_splitnodeFeature = args[9]
    thres_coeff_var = args[10]
    name_run = ""
    if ml_model == "RF":
        tree_method = args[0]
        k = args[1]
        ntrees = args[2]
        if prior_file == "no":
            name_run = datasetname+"_output_"+tree_method+"_K"+k+"_ntrees"+str(ntrees)+"_h"+str(timehorizon)+"_datatype"+str(datatype)+"_bias_score_splitnodeFeature"+str(bias_score_splitnodeFeature)
        else:
            name_run = datasetname+"_output_"+tree_method+"_K"+k+"_ntrees"+str(ntrees)+"_h"+str(timehorizon)+"_datatype"+str(datatype)+"_bias_score_splitnodeFeature"+str(bias_score_splitnodeFeature)+"_prior"+((prior_file).split('.')[0])

    #script_filename = inspect.getfile(inspect.currentframe()) # script filename (usually with path)
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory

    print "loading Dataset ", datasetname
    ds1 = Dataset(datasetname)
    start_time = timeit.default_timer()

    dict_X_train_ts, dict_y_train_ts, dict_X_test_ts, dict_y_test_ts, dict_X_train_ss, \
    dict_y_train_ss, dict_X_test_ss, dict_y_test_ss, genelist, tflist, goldstandard, \
    output_path \
        = ds1.loadDataset(script_dir, timehorizon, test_size, prior_type, prior_file, thres_coeff_var, name_run)

    elapsed_time = timeit.default_timer() - start_time
    print ("Elapsed %s" % (elapsed_time))

    path_dataset = script_dir+"/Datasets/"+datasetname+"/"

    #plt.switch_backend(u'MacOSX')
    sum_m_err, td_var = 0, 0
    for i, k in enumerate(dict_X_train_ss.keys()):
        # if i > 4:
        #     break
        regr = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=ntrees)
        X, y, Xt, yt = dict_X_train_ss.get(k), dict_y_train_ss.get(k), \
                       dict_X_test_ss.get(k), dict_y_test_ss.get(k)
        regr.fit(X, y)
        y_pred = regr.predict(Xt)
        # ax = plt.subplot(1, 5, i+1)
        # ax.scatter(Xt[:,1], y_pred, c='r')
        # ax.scatter(Xt[:,1], yt, c='b')
        score = regr.score(Xt, yt)
        # ax.text(5, 10, ('Score: {}'.format(score)), size = 15, horizontalalignment = 'right')
        imp_features = regr.feature_importances_
        most_imp, top3_imp = np.argmax(imp_features), imp_features.argsort()[-3:][::-1]
        m_err, yt_var = mse_err(y_pred, yt)/len(yt), np.var(yt)
        sum_m_err += m_err
        td_var += yt_var
        print('{}, Most Important Index: {}, Top 3 Important Indices: {}, MSE: {}, Data Var: {}'.format(
            k, genelist[most_imp], genelist[top3_imp], m_err, yt_var))
    # plt.tight_layout()
    # plt.show()
    print ('Avg Mean Sq Error: {}, Test Data Variance: {}'.format(sum_m_err/len(genelist), td_var/len(genelist)))
    print ("Model parameters: {}".format(regr.get_params()))

if __name__ == '__main__':
    parser = argp.ArgumentParser()
    parser.add_argument('-tm', '--tree_method', type=str, help='Tree-based method: RF-Random Forest or ET-Extra Trees)', default='RF')
    parser.add_argument("-k", "--K", type=str, help="The max num of features used for node splitting", default='sqrt')
    parser.add_argument("-nt", "--ntrees", type=int, help="The number of trees", default=500)#200)
    parser.add_argument("-th", "--timehorizon", type=int, help="The time-lag h", default=1)
    parser.add_argument("-ds", "--datasetname", type=str, help="The dataset name. e.g. bsubtilis", default='dream4')#'Arabidopsis_Canal')#default='dream10_debug')bsubtilis#Arabidopsis_Amys_root_allTFs
    parser.add_argument("-ts", "--test_size", type=float, help="The size of the dataset to use as test set", default=0.15)#0.15)
    parser.add_argument("-dt", "--data_type", type=str, help="Type of Data: TS, SS or TS-SS", default="SS")
    parser.add_argument("-pf", "--prior_file", type=str, help="Prior file that will be read from directory datasetname/Priors", default="no")#"_ranking_780_genes_SS.txt")#="no")#"gold_standard.txt")#"_ranking_10_genes_SS.txt")#default="_ranking_10_genes_TS.txt")#default="_ranking_10_genes_SS.txt")#default="gold_standard.txt")#default="no")
    parser.add_argument("-pt", "--prior_type", type=str, help="Type of Weights: binary_all (e.g. gold standard prior), real_all (e.g. steady state prior) values", default="no")#"real_all")#"binary_all")#"real_all")#default="real_all")#default="binary_all")#default="no")
    # Format "real_all": list of edges with weights ; Format "binary_all" : adjacency matrix
    #"_all" stands for: the priors file contain complete information from all the TFs in the universe to all the target genes in the universe
    #an example of binary is goldstandard with 0 and 1
    #an example of real is weights coming from run of RF with steady state data 
    parser.add_argument("-snf", "--bias_score_splitnodeFeature", type=str, help="Whether to use the prior weights to bias the score of the split node features candidates", required=False, default="yes")#default="")

    #0 thres means no filtering
    parser.add_argument("-thres", "--thres_coeff_var", type=float, help="The coefficient of variation cutoff", default=0)#1.6)#0.141)#0.05)#0.25)#0.06)#0.1) #0.2ecoli #2dream4 #0.25bsubtilis

    args = parser.parse_args()
    
    print args.tree_method, "max_feat", args.K, "ntrees", args.ntrees, args.timehorizon, args.datasetname, args.test_size, args.prior_file, args.prior_type, args.data_type, args.bias_score_splitnodeFeature, args.thres_coeff_var
    main("RF", args.tree_method, args.K, args.ntrees, args.timehorizon, args.datasetname, args.test_size, args.prior_file, args.prior_type, args.data_type, args.bias_score_splitnodeFeature, args.thres_coeff_var)


