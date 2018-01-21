import matplotlib.pyplot as plt
import numpy as np
import argparse as argp

def sq_err(a, b):
    assert(a.shape == b.shape)
    return (sum(np.square(a - b)))

def mse_err(a, b):
    return sq_err(a, b) * 1.0 / a.shape[0]

import heapq
def items_by_sqe(x, items):
    items_heap = []
    for i, item in enumerate(items):
        if sum(item) == 0:
            continue
        heapq.heappush(items_heap, (sq_err(x, item), i))
    return items_heap

import pandas as pd
def read_genelist():
    return pd.DataFrame.from_csv('./data/genelist.csv')

def read_csv(item):
    df_train = pd.DataFrame.from_csv('./data/' + item + '_Train_SteadyState.csv')
    df_test = pd.DataFrame.from_csv('./data/' + item + '_Test_SteadyState.csv')
    return df_train, df_test

def save_top3(df_plot, y_pred, indices, gene, train_test='train'):
    fig = plt.figure()
    ax = df_plot.iloc[:, np.append(indices, 100)].plot() #100 is yt, the correct value
    if y_pred is not None:
        ax.plot(y_pred, label='Predicted')
    plt.savefig('./exp_results/figures/dcsn_tree/' + train_test + '/' + gene + '_top3.png', bbox_inches='tight')
    plt.close(fig)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
def main(ml_model, *args):
    #Turn off interactive plotting
    plt.ioff()
    d_mx, d_mn = -1, 1e9
    sum_m_err, td_var = 0, 0
    df_genelist = read_genelist()
    for i, k in df_genelist.iterrows():
        curr_gene = k.values[0]
        df_train, df_test = read_csv(k[0])
        X, y = df_train.iloc[:, 0:100].values, df_train.iloc[:, 100].values
        Xt, yt = df_test.iloc[:,0:100].values, df_test.iloc[:, 100].values
        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7), n_estimators=500)

        d_mx, d_mn = max(d_mx, np.max(X)), min(d_mn, np.min(X))
        d_mx, d_mn = max(d_mx, np.max(Xt)), min(d_mn, np.min(Xt))

        regr.fit(X, y)
        y_pred = regr.predict(Xt)
        top3_by_imp = np.argsort(regr.feature_importances_)[::-1][0:3]
        #top3_by_sqe = np.argsort(items_by_sqe(y_pred, Xt.T))[::-1][0:3]
        most_imp = top3_by_imp[0]
        m_err, yt_var = mse_err(y_pred, yt), np.var(yt)
        sum_m_err += m_err
        td_var += yt_var
        save_top3(df_train, None, top3_by_imp, curr_gene, 'train')
        save_top3(df_test, y_pred, top3_by_imp, curr_gene, 'test')
        print('{}, Most Important Index: {}, Top 3 Important Indices: {}, MSE: {}, Data Var: {}'.format(
            curr_gene, df_genelist.iloc[most_imp].values[0], df_genelist.iloc[top3_by_imp].values[:,0], m_err, yt_var))
    print ('Avg Mean Sq Error: {}, Test Data Variance: {}'.format(sum_m_err/100, td_var/100))
    print ("Model parameters: {}".format(regr.get_params()))
    print("Max: {}, Min: {}".format(d_mx, d_mn))



if __name__ == '__main__':
    parser = argp.ArgumentParser()
    parser.add_argument('-tm', '--tree_method', type=str, help='Tree-based method: RF-Random Forest or ET-Extra Trees)', default='RF')
    parser.add_argument("-k", "--K", type=str, help="The max num of features used for node splitting", default='sqrt')
    parser.add_argument("-nt", "--ntrees", type=int, help="The number of trees", default=1)#200)
    parser.add_argument("-th", "--timehorizon", type=int, help="The time-lag h", default=1)
    parser.add_argument("-ds", "--datasetname", type=str, help="The dataset name. e.g. bsubtilis", default='dream4')#'Arabidopsis_Canal')#default='dream10_debug')bsubtilis#Arabidopsis_Amys_root_allTFs
    parser.add_argument("-ts", "--test_size", type=float, help="The size of the dataset to use as test set", default=0.15)#0.15)
    parser.add_argument("-dt", "--data_type", type=str, help="Type of Data: TS, SS or TS-SS", default="SS")
    parser.add_argument("-pf", "--prior_file", type=str, help="Prior file that will be read from directory datasetname/Priors", default="no")#"_ranking_780_genes_SS.txt")#="no")#"gold_standard.txt")#"_ranking_10_genes_SS.txt")#default="_ranking_10_genes_TS.txt")#default="_ranking_10_genes_SS.txt")#default="gold_standard.txt")#default="no")
    parser.add_argument("-pt", "--prior_type", type=str, help="Type of Weights: binary_all (e.g. gold standard prior), real_all (e.g. steady state prior) values", default="no")#"real_all")#"binary_all")#"real_all")#default="real_all")#default="binary_all")#default="no")
    parser.add_argument("-snf", "--bias_score_splitnodeFeature", type=str, help="Whether to use the prior weights to bias the score of the split node features candidates", required=False, default="yes")#default="")

    #0 thres means no filtering
    parser.add_argument("-thres", "--thres_coeff_var", type=float, help="The coefficient of variation cutoff", default=0)#1.6)#0.141)#0.05)#0.25)#0.06)#0.1) #0.2ecoli #2dream4 #0.25bsubtilis

    args = parser.parse_args()
    
    print args.tree_method, "max_feat", args.K, "ntrees", args.ntrees, args.timehorizon, args.datasetname, args.test_size, args.prior_file, args.prior_type, args.data_type, args.bias_score_splitnodeFeature, args.thres_coeff_var
    main("RF", args.tree_method, args.K, args.ntrees, args.timehorizon, args.datasetname, args.test_size, args.prior_file, args.prior_type, args.data_type, args.bias_score_splitnodeFeature, args.thres_coeff_var)


