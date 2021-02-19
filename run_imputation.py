import pandas
import from sklearn linear_model

from sklearn.model_selection import KFold
from sklearn import metrics

print("Hello")

dataset_missing = pandas.read_csv("dataset_missing.csv", dtype='object')

print(dataset_missing)

#Detecting missing data
print(dataset_missing.isnull().sum()) #when data is missing, true , when not, false. sum

#5% missing, not so bad . Good use this method until you have about 10% 
#earning = many that determine learning.... But, here we are not so worried about that 
#We don't need to deal with endogeneity, etc. We are just trying to fill the missing values. We are just trying to predict. 
# Best guess in earnings. We are caring about the correlation not prediction. 

#Examining the distribution of the missing data
# For example, according to year, or to id

for i in range(2000,2011):
	print("year", str(i))
	print(dataset_missing[dataset_missing['year']==str(i)].isnull().sum()) #== means that this is a condition. 
	
def data_imputation(dataset, impute_target_name, impute_data) #We are defining the function where we will 
	impute_target = dataset[impute_target_name]
	sub_dataset = pandas.concat([impute_target, impute_data], axis = 1) 
	#concat is to merge two datasets together. We use the imput target and te impute data
	#axis =1 is horizontally merging the data
	data = sub_dataset.loc[:,sub_dataset.columns != impute_target_name][sub_dataset[impute_target_name].notnull()] #   != means that it is not equal to the value, while == is when you are defining
	target = dataset[impute_target_name][sub_dataset[impute_target_name].notnull()].values #earning, the one that is missing

	kfold_object = KFold(n_splits=4)
	kfold_object.get_n_splits(data)


	i=0
	for training_index, test_index in kfold_object.split(data):#4 cases
		i=i+1
		print("case: ", str(i))
		data_training = data[training_index]
		data_test = data[test_index]
		target_training = target[training_index]
		target_test = target[test_index]
		one_fold_machine = linear_model.LinearRegression()
		one_fold_machine.fit(data_training, target_training)
		new_target = one_fold_machine.predict(data_test)
		print(metrics.mean_absolute_error(target_test, new_target))



	machine = linear_model.LinearRegression()
	machine.fit(data, target)
	return machine.predict(sub_dataset.loc[:,sub_dataset.columns!=impute_target_name],values)
	#you can store the value 

impute_data = dataset_missing[["ability","age","female","education","exp"]]
# Is there a way to not need two sets of []? 
results = data_imputation(dataset_missing, "earnings", impute_data) # we create the impute data

#however, be careful with categorical variables (like occupation, )

#To form dummies for categorical variables (obs: there is not a i.regions like in stata)

region_dummies = pandas.get_dummies(dataset_missing["region"])
occ_dummies = pandas.get_dummies(dataset_missing["occ"])

#array vs. 

impute_data = pandas.concat([dataset_missing[["ability","age","female","education"]]])
#concat means that we want to put two things together

#Horizontally merging your data : axis = 1

#Hash of stuff,
new_earnings = pandas.concat([dataset_missing, new_earnings], axis=1)
#new_earnings = new_earnings.rename(columns={0:"earning_imputed"})#obs: if you want to rename multiple columns , just add a "," and do others
new_earnings.rename(columns={0:"earning_imputed"}, inplace=True)#obs: if you want to rename multiple columns , just add a "," and do others

print(new_earnings)

dataset_missing = pandas.concat([data_missing, new_earnings], axis=1)

#Now, fill the earnings with the earnings_imputed where data is missing for earnings. Instead of just using earnings_imputed in the place of 

dataset_missing['earnings_missing'] = dataset_missing['earnings'].isnull()

print(dataset_missing)
print(dataset_missing.isnull(),sum())

dataset_missing['earnings'].fillna(dataset_missing['earnings_imputed'], inplace=True) #inplace is faster than defining again dataset_missing = ....

print(dataset_missing.isnull(),sum())

dataset_missing.drop(columns=["earnings_imputed","earnings_missing"], inplace=True)

#print = dataset_missing.earnings.round()# wrong. But there is a way to round to the nearest integer or decimal point. There is a dedicated function on pandas for that, which is quicker. 

dataset_missing['earnings'] = dataset_missing['earnings'].astype(float).round(2)
data_missing.to_csv("data_not_missing.csv", index=False)








