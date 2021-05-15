import csv
import pandas as pd
import numpy as np
import os
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import scipy.io as io
import datetime

def ReadFile(filepath):
	tmp_lst = []
	with open(filepath, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			tmp_lst.append(row)
	df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])
	return df

def getfrt(df):
	df['dayofweek'] = df['date'].dt.dayofweek+1
	df['day'] = df['date'].dt.day
	df['month'] = df['date'].dt.month
	df['year'] = df['date'].dt.year
	return df

def Preprocessing(filepath, train=True):
	if train:
		df1 = ReadFile(filepath+'\\train_v2.csv')
		df1 = df1.astype({'amount':'float32'})

	else:
		df1 = ReadFile(filepath+'\\test_v2_periods.csv')
		df1 = df1[['date','post_id','periods']]
	df1 = df1.astype({'periods':'float32'})
	df2 = ReadFile(filepath+'\\wkd_v1.csv')
	drop_date = io.loadmat('drop.mat')['my_drop']
	for ii in range(len(drop_date)):
		#datetimeout = datetime.datetime.strptime(drop_date[ii],'%Y/%m/%d')
		df1[df1['date']==drop_date[ii]] = np.nan
	df1.dropna(how='any',axis=0,inplace=True)
	df1['date'] = pd.to_datetime(df1['date'])
	df2class = pd.get_dummies(df2['WKD_TYP_CD'])
	df2 = pd.concat([df2['ORIG_DT'], df2class], axis=1)
	df2['ORIG_DT'] = pd.to_datetime(df2['ORIG_DT'])
	df2 = df2.rename(columns={'ORIG_DT':'date'})
	dfcombine = pd.merge(df1,df2)
	dfcombine = getfrt(dfcombine)
	return dfcombine

def lgb_cv(train_x, train_y, test_x):
    predictors = list(train_x.columns)
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train = np.zeros((train_x.shape[0]))
    test = np.zeros((test_x.shape[0]))
    test_pre = np.zeros((folds, test_x.shape[0]))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    tpr_scores = []
    cv_rounds = []
    
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        train_matrix = lightgbm.Dataset(tr_x, label=tr_y)
        test_matrix = lightgbm.Dataset(te_x, label=te_y)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metrics':'mean_squared_error',
            'num_leaves': 2 ** 6-1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'learning_rate': 0.05,
            'seed': 2021,
            'nthread': 8,
            'verbose': -1,
        }
        num_round = 4000
        early_stopping_rounds = 100
        if test_matrix:
            model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=200,
                              #feval=tpr_eval_score,
                              early_stopping_rounds=early_stopping_rounds
                              )
            print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))[:10]))
            importance_list=[ x[0] for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))]
            #print(importance_list)
            pre = model.predict(te_x, num_iteration=model.best_iteration)#
            pred = model.predict(test_x, num_iteration=model.best_iteration)#
            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(mean_squared_error (te_y, pre))
            cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = pred
        #
        print("cv_score is:", cv_scores)
    use_mean=True
    if use_mean:
        test[:] = test_pre.mean(axis=0)
    else:
        pass
    #
    print("val_mean:" , np.mean(cv_scores))
    print("val_std:", np.std(cv_scores))
    return train, test, test_pre_all, np.mean(cv_scores),importance_list

def xgb_cv(train_x, train_y, test_x):
    predictors = list(train_x.columns)
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    folds = 10
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train = np.zeros((train_x.shape[0]))
    test = np.zeros((test_x.shape[0]))
    test_pre = np.zeros((folds, test_x.shape[0]))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    tpr_scores = []
    cv_rounds = []
    
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        train_matrix = xgb.DMatrix(tr_x, label=tr_y)
        test_matrix = xgb.DMatrix(te_x, label=te_y)
        params = {
            'booster':'gbtree',
            'objective':'reg:linear',
            'gamma':0,
            'max_depth':6,
            'lambda':2,
            'subsample':1,
            'colsample_bytree':1,
            'min_child_weight':1,
            'slient':1,
            'eta':0.01,
            'seed':2021,
            'eval_metric':'rmse'
            }
        num_round = 4000
        early_stopping_rounds = 100
        evals = [(train_matrix,'train'),(test_matrix,'val')]
        if test_matrix:
            model = xgb.train(params, train_matrix, num_round, evals, obj=None, feval=None, maximize='False',
                              #feval=tpr_eval_score,
                              early_stopping_rounds=early_stopping_rounds, evals_result=None, verbose_eval=200
                              )

            #feature_importance = model.get_score(importance_type='gain')
            #feature_importance_list = list(feature_importance.values())
            #print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, feature_importance),
            #            key=lambda x: x[1],reverse=True))[:10]))
            #importance_list=[ x[0] for x in list(sorted(zip(predictors, feature_importance),
            #            key=lambda x: x[1],reverse=True))]
            #print(importance_list)
            te_x_in = xgb.DMatrix(te_x)
            test_x_in = xgb.DMatrix(test_x)
            pre = model.predict(te_x_in, ntree_limit=model.best_iteration)#
            pred = model.predict(test_x_in, ntree_limit=model.best_iteration)#
            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(mean_squared_error (te_y, pre))
            cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = pred
        #
        print("cv_score is:", cv_scores)
    use_mean=True
    if use_mean:
        test[:] = test_pre.mean(axis=0)
    else:
        pass
    #
    print("val_mean:" , np.mean(cv_scores))
    print("val_std:", np.std(cv_scores))
    return train, test, test_pre_all, np.mean(cv_scores) 	

if __name__=="__main__":
	dftrain = Preprocessing('L:\\zhaohangContestv2',train=True)
	dftest = Preprocessing('L:\\zhaohangContestv2',train=False)
	A_index = ['A'+str(i) for i in range(1,14)]
	B_index = 'B1'
	testsub_A = dftest[dftest['post_id']=='A']
	testsubin_A = testsub_A.drop(['date','post_id'],axis=1)
	testsub_B = dftest[dftest['post_id']=='B']
	testsubin_B = testsub_B.drop(['date','post_id'],axis=1)

	output_A = np.zeros([1,testsubin_A.shape[0]])
	output_B = np.zeros([1,testsubin_B.shape[0]])

	# prediction for A
	dftrain = dftrain[(dftrain['year']==2020)&(dftrain['month']>5)].reset_index(drop=True)
	for ii, Astr in enumerate(A_index):
		trainsub = dftrain[dftrain['biz_type']==Astr]
		trainsubY = trainsub[['amount']]/1e4
		trainsubX = trainsub.drop(['date','biz_type','post_id','amount'],axis=1)
		print(trainsubX.shape, trainsubY.shape, testsubin_A.shape)
		lgb_train, lgb_test, sb, cv_scores, _ = lgb_cv(trainsubX, trainsubY, testsubin_A)
		lgb_test_A=[item if item>0 else 0 for item in lgb_test]
		output_A = output_A + np.array(lgb_test_A)

    # prdiction for B
	trainsub = dftrain[dftrain['biz_type']==B_index]
	trainsubY = trainsub[['amount']]/1e4
	trainsubX = trainsub.drop(['date','biz_type','post_id','amount'],axis=1)
	lgb_train, lgb_test, sb, cv_scores, _ = lgb_cv(trainsubX, trainsubY, testsubin_B)
	lgb_test_B=[item if item>0 else 0 for item in lgb_test]
	output_B = np.array(lgb_test_B)


	output_A = output_A*1e4
	output_B = output_B*1e4
	output_A = output_A.astype('int32')
	output_B = output_B.astype('int32')
	output_A = output_A.reshape(-1)
	output_B = output_B.reshape(-1)
	#output_A = np.where(output_A>0,output_A,0)
	#output_B = np.where(output_B>0,output_B,0)

	test_day=ReadFile('test_v2_periods.csv')#按天计算
	pre_day=[]
	for i in range(31):
		for j in range(48):
			if j <= 15 or j>=36:
				pre_day.append(0)
			else:
				pre_day.append(output_A[48*i+j])
		for j in range(48):
			if j<=15 or j>=38:
				pre_day.append(0)
			else:
				pre_day.append(output_B[48*i+j])
	test_day['amount']=pre_day
#
	if not os.path.exists('L:\\zhaohangContestv2\\submitTreenew'):
		os.makedirs('L:\\zhaohangContestv2\\submitTreenew')
	f=open('L:\\zhaohangContestv2\\submitTreenew\\test_day_period.txt','w')
	f.write('Date'+','+'Post_id'+','+'Periods'+','+'Predict_amount'+'\n')
	for _,date,post_id,periods,amount in test_day.itertuples():
		f.write(date+','+post_id+','+str(int(periods))+','+str(int(amount))+'\n')
	f.close()

