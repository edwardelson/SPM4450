# Logistic Regression inspired by https://www.kaggle.com/xenocide/shelter-animal-outcomes/shelter-animal-random-forest
# try various breed, color (correlation), outcomeSubtype, don't split into various binaries axis

# instead of factorization, i could have used sklearn.preprocessing.labelencoder

import numpy as np
import pandas as pd 

#import data
shelter_train = pd.read_csv("train.csv")
shelter_test = pd.read_csv("test.csv")

#shelter_train.dropna(axis=0)

#separate outcome variable from train, for prediction
train_outcome = shelter_train["OutcomeType"]
#collect AnimalID, for submission
train_ID = shelter_train["AnimalID"]
test_ID = shelter_test["ID"]

#########################################################################################################
####Pre-Processing, mostly dropping and altering data
#########################################################################################################

#find year from date
time_train = pd.to_datetime(shelter_train["DateTime"])
time_test = pd.to_datetime(shelter_test["DateTime"])

shelter_train["Year"] = time_train.dt.year
shelter_test["Year"] = time_test.dt.year
shelter_train["Month"] = time_train.dt.month
shelter_test["Month"] = time_test.dt.month
shelter_test["Day"] = time_test.dt.day
shelter_train["Day"] = time_train.dt.day
shelter_test["Hour"] = time_test.dt.hour
shelter_train["Hour"] = time_train.dt.hour
shelter_test["Minute"] = time_test.dt.minute
shelter_train["Minute"] = time_train.dt.minute


#replace Season with integers
intval, label = pd.factorize(shelter_train["Month"], sort=True)
shelter_train["Month"] = pd.DataFrame(intval)
del intval, label
intval, label = pd.factorize(shelter_test["Month"], sort=True)
shelter_test["Month"] = pd.DataFrame(intval)
del intval, label
#drop DateTime
shelter_train.drop("DateTime", axis=1, inplace=True)
shelter_test.drop("DateTime", axis=1, inplace=True)

# #convert date to Season
# def season_group(date):
# 	try:
# 		date_list = date.split() #"2 days" -> ["2", "days"]
# 	except:
# 		return None
# 	month = int(date_list[0][0:4])	#0:4-> year, 5:7 -> month
# 	return month	
# #replace DateTime with Season
# shelter_train["DateTime"] = shelter_train["DateTime"].apply(season_group)
# shelter_test["DateTime"] = shelter_test["DateTime"].apply(season_group)
# #replace Season with integers
# intval, label = pd.factorize(shelter_train["DateTime"], sort=True)
# shelter_train["DateTime"] = pd.DataFrame(intval)
# del intval, label
# intval, label = pd.factorize(shelter_test["DateTime"], sort=True)
# shelter_test["DateTime"] = pd.DataFrame(intval)
# del intval, label

shelter_train["SexuponOutcome"].fillna("Spayed Female", inplace=True)
shelter_test["SexuponOutcome"].fillna("Spayed Female", inplace=True)

#Sex Intact
def intact_group(sex):
	try:
		intact_type = sex.split()
	except:
		return 0
	if intact_type[0] == "Neutered" or intact_type[0] ==  "Spayed":		
		return 1
	elif intact_type[0] == "Intact":
		return 2
	else:
		return 0

#create Sex Intact / Virginity
shelter_train["Virginity"] = shelter_train["SexuponOutcome"].apply(intact_group)
shelter_test["Virginity"] = shelter_test["SexuponOutcome"].apply(intact_group)

#Sex
def sex_group(sexs):
	try:
		sex_type = sexs.split()
	except:
		return 0
	#categorize
	if sex_type[0] == "Unknown":
		return 0
	elif sex_type[1] == "Male":
		return 1
	elif sex_type[1] == "Female":
		return 2
	else:
		return 0

#create Sex Intact / Virginity
shelter_train["Sex"] = shelter_train["SexuponOutcome"].apply(sex_group)
shelter_test["Sex"] = shelter_test["SexuponOutcome"].apply(sex_group)
# intval, label = pd.factorize(shelter_train["SexuponOutcome"], sort=True)
# shelter_train["SexuponOutcome"] = pd.DataFrame(intval)
# del intval, label
# intval, label = pd.factorize(shelter_test["SexuponOutcome"], sort=True)
# shelter_test["SexuponOutcome"] = pd.DataFrame(intval)
# del intval, label
#Drop Sex
shelter_train.drop("SexuponOutcome", axis=1, inplace=True)
shelter_test.drop("SexuponOutcome", axis=1, inplace=True)

#AnimalType
intval, label = pd.factorize(shelter_train["AnimalType"], sort=True)
shelter_train["AnimalType"] = pd.DataFrame(intval)
del intval, label
intval, label = pd.factorize(shelter_test["AnimalType"], sort=True)
shelter_test["AnimalType"] = pd.DataFrame(intval)
del intval, label


#Name: author thinks that animals with names, have higher change to be adopted
def check_has_name(name):
	if type(name) is str:
		return 1
	else: #if name is NaN
		return 0

#has_name: create new column for has_name
shelter_train["has_name"] = shelter_train["Name"].apply(check_has_name)
shelter_test["has_name"] = shelter_test["Name"].apply(check_has_name)
#drop the name column
shelter_train.drop("Name", axis=1, inplace=True)
shelter_test.drop("Name", axis=1, inplace=True)

#Age: fill missing data in Age, assume NaN = 1 year (modus value)
shelter_train["AgeuponOutcome"].fillna("1 year", inplace=True)
shelter_test["AgeuponOutcome"].fillna("1 year", inplace=True)
#convert age to age group, author arbitrarily decides
def age_group(age):
	try:
		age_list = age.split() #"2 days" -> ["2", "days"]
	except:
		return None
	ages = int(age_list[0])
	if(age_list[1].find("s")): #weeks->week, days->day
		age_list[1] = age_list[1].replace("s","")
	if age_list[1] == "day":
		return ages
	elif (age_list[1] == "week"):
		return ages*7
	elif (age_list[1] == "month"):
		return ages*30
	elif (age_list[1] == "year"):
		return ages*365

	# elif (int(age_list[0]) >= 1 and int(age_list[0]) <= 4) and (age_list[1] == "month"):
	# 	return int(age_list[0])*30 #"early_young" 
	# elif (int(age_list[0]) >= 5 and int(age_list[0]) <= 12) and (age_list[1] == "month"):
	# 	return int(age_list[0])*30 #"late_young"
	# elif (int(age_list[0]) >= 1 and int(age_list[0]) <= 3) and (age_list[1] == "year"):
	# 	return int(age_list[0])*365 #"early_adult" #between 1 and 10 years -> Adult
	# elif (int(age_list[0]) >= 3 and int(age_list[0]) <= 5) and (age_list[1] == "year"):
	# 	return int(age_list[0])*365 #"late_adult" #between 1 and 10 years -> Adult
	# elif (int(age_list[0]) >= 6 and int(age_list[0]) <= 8) and (age_list[1] == "year"):
	# 	return int(age_list[0])*365 #"early_senior" #between 1 and 10 years -> Adult
	# else:
	# 	return int(age_list[0])*365 #"late_senior" #above 10 years	 -> Senior

#replace AgeuponOutcome with Age group
shelter_train["AgeuponOutcome"] = shelter_train["AgeuponOutcome"].apply(age_group)
shelter_test["AgeuponOutcome"] = shelter_test["AgeuponOutcome"].apply(age_group)
# #### convert each unique label to unique integers
# intval, label = pd.factorize(shelter_train["AgeuponOutcome"], sort=True)
# shelter_train["AgeuponOutcome"] = pd.DataFrame(intval)
# del intval, label
# intval, label = pd.factorize(shelter_test["AgeuponOutcome"], sort=True)
# shelter_test["AgeuponOutcome"] = pd.DataFrame(intval)
# del intval, label
# shelter_train.drop("AgeuponOutcome", axis=1, inplace=True)
# shelter_test.drop("AgeuponOutcome", axis=1, inplace=True)

#ID: Drop
shelter_train.drop("AnimalID", axis=1, inplace=True)
shelter_test.drop("ID", axis=1, inplace=True)

#aggressiveness based on breed type. Most dangerous breeds:
# Pitbull (55-65 lbs), Rottweiler (100-130 lbs), Husky-type (66 lbs), German
# Shepherd (100 lbs) , Alaskan Malamute (100 lbs), Doberman pinscher (65-90lbs),
# chow chow (70 lbs), Great Danes (200pounds), Boxer (70 lbs), Akita (45 kg)
def aggressive(breed):
	if breed.find("Pit Bull") != -1:
		return 1
	elif breed.find("Rottweiler") != -1:
		return 2#1
	elif breed.find("Husky") != -1:
		return 3#1
	elif breed.find("Shepherd") != -1:
		return 4#1
	elif breed.find("Malamute") != -1:
		return 5#1
	elif breed.find("Doberman") != -1:
		return 6#1
	elif breed.find("Chow") != -1:
		return 7#1
	elif breed.find("Dane") != -1:
		return 8#1
	elif breed.find("Boxer") != -1:
		return 9#1
	elif breed.find("Akita") != -1:
		return 10#1
	else:
		return 11#2

shelter_train["Aggresiveness"] = shelter_train["Breed"].apply(aggressive)
shelter_test["Aggresiveness"] = shelter_test["Breed"].apply(aggressive)

#Most allergic breeds:
#Akita, Alaskan Malamute, American Eskimo, Corgi, Chow-chow, German
#Shepherd, Great Pyrenees, Labrador, Retriever, Husky
def allergic(breed):
	if breed.find("Akita") != -1:
		return 1
	elif breed.find("Malamute") != -1:
		return 2#1
	elif breed.find("Eskimo") != -1:
		return 3#1
	elif breed.find("Corgi") != -1:
		return 4#1
	elif breed.find("Chow") != -1:
		return 5#1
	elif breed.find("Shepherd") != -1:
		return 6#1
	elif breed.find("Pyrenees") != -1:
		return 7#1
	elif breed.find("Labrador") != -1:
		return 8#1
	elif breed.find("Retriever") != -1:
		return 9#1
	elif breed.find("Husky") != -1:
		return 10#1
	else:
		return 11#2

shelter_train["Allergic"] = shelter_train["Breed"].apply(allergic)
shelter_test["Allergic"] = shelter_test["Breed"].apply(allergic)

#weight based on breed type. Most dangerous breeds:
# Below 100 lbs: Pitbull (55-65 lbs), Husky-type (66 lbs), Doberman pinscher (65-90lbs), Boxer (70 lbs), Akita (45 kg), chow chow (70 lbs)
# Above 100 lbs: Rottweiler (100-130 lbs), German Shepherd (100 lbs), Alaskan Malamute (100 lbs), Great Danes (200pounds), 

def weight(breed):
	if breed.find("Pit Bull") != -1:
		return 1
	elif breed.find("Husky") != -1:
		return 1
	elif breed.find("Doberman") != -1:
		return 1
	elif breed.find("Boxer") != -1:
		return 1
	elif breed.find("Akita") != -1:
		return 1
	elif breed.find("Chow") != -1:
		return 1
	elif breed.find("Rottweiler") != -1:
		return 2
	elif breed.find("Shepherd") != -1:
		return 2
	elif breed.find("Malamute") != -1:
		return 2
	elif breed.find("Dane") != -1:
		return 2
	else:
		return 2

shelter_train["Weight"] = shelter_train["Breed"].apply(weight)
shelter_test["Weight"] = shelter_test["Breed"].apply(weight)

#fetch breed type
def breed_group(breed_input):
	breed = str(breed_input)
	if (' ' in breed) == False:
		return breed #only 1 word
	breed_list = breed.split()
	try:
		return breed_list[2] #fetch last word, for 1 words breed
	except:
		return breed_list[1] #fetch last word, for 2 words breed
	return breed

shelter_train["Breed"] = shelter_train["Breed"].apply(breed_group)
shelter_test["Breed"] = shelter_test["Breed"].apply(breed_group)

#### convert each unique label to unique integers
intval, label = pd.factorize(shelter_train["Breed"], sort=True)
shelter_train["Breed"] = pd.DataFrame(intval)
del intval, label
intval, label = pd.factorize(shelter_test["Breed"], sort=True)
shelter_test["Breed"] = pd.DataFrame(intval)
del intval, label
# #Breed: Drop
# shelter_train.drop("Breed", axis=1, inplace=True)
# shelter_test.drop("Breed", axis=1, inplace=True)

# Color Intact
def color_group(color):
	try:
		color_type = color.split()
	except:
		return "unknown"
	return str(color_type[0])

#create color
shelter_train["Color"] = shelter_train["Color"].apply(color_group)
shelter_test["Color"] = shelter_test["Color"].apply(color_group)
#### convert each unique label to unique integers
intval, label = pd.factorize(shelter_train["Color"], sort=True)
shelter_train["Color"] = pd.DataFrame(intval)
del intval, label
intval, label = pd.factorize(shelter_test["Color"], sort=True)
shelter_test["Color"] = pd.DataFrame(intval)
del intval, label
# #Color: Drop
# shelter_train.drop("Color", axis=1, inplace=True)
# shelter_test.drop("Color", axis=1, inplace=True)

#OutcomeType: Drop
shelter_train.drop("OutcomeType", axis=1, inplace=True)

#OutcomeSubType: Drop
shelter_train.drop("OutcomeSubtype", axis=1, inplace=True)

print(shelter_train.head())

#########################################################################################################
#### PCA
#########################################################################################################
# from sklearn.decomposition import PCA
# n = 14
# pca_train = PCA(n_components=n)
# pca_test = PCA(n_components=n)
# pca_train.fit(shelter_train)
# pca_test.fit(shelter_test)

# shelter_train = pd.DataFrame(pca_train.transform(shelter_train))
# shelter_test = pd.DataFrame(pca_test.transform(shelter_test))

#########################################################################################################
#### Build Models
#########################################################################################################
# #both will be merged to all_encoded, need to tag train data with 1 and test with 0
# shelter_train["train"] = 1
# shelter_test["train"] = 0

# #vector all consists all train and test data
# all = pd.concat([shelter_train, shelter_test])
# #different values from column are separated into another columns, many dimensions (each binary)
# all_encoded = pd.get_dummies(all, columns = ['train'])

# #fetch shelter_train and shelter_test
# model_train = all_encoded[all_encoded["train_1"]==1]
# model_test = all_encoded[all_encoded["train_0"]==1]
# #drop train_0 and train_1 columns
# model_train.drop(["train_0","train_1"], axis=1, inplace=True)
# model_test.drop(["train_0","train_1"], axis=1, inplace=True)

model_train = shelter_train
model_test = shelter_test

#split test and train randomly
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(model_train, train_outcome, test_size=0.3)

# #split test fixed
# X_train = model_train[0:20000]
# y_train = train_outcome[0:20000]
# X_val = model_train[20000:26729]
# y_val = train_outcome[20000:26729]

#########################################################################################################
#### Prediction Model
#########################################################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    # KNeighborsClassifier(1000),
    # SVC(max_iter=100, probability=True, kernel='rbf', degree=10),
    # SVC(gamma=2, C=1),
    # DecisionTreeClassifier(max_depth=3),
    # RandomForestClassifier(max_depth=5, n_estimators=500, max_features=1),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    GradientBoostingClassifier(learning_rate=0.05, n_estimators = 500, min_samples_leaf=5)
    ]

for classifier in classifiers:
	log = classifier
	log.fit(X_train, y_train)

	# log knows how many classes are there in y_train
	y_probs = log.predict_proba(X_val)
	y_pred = log.predict(X_val)

	#this columns' order can be found by "log.classes_"
	new_frame = pd.DataFrame(y_probs, columns=["Adoption","Died","Euthanasia","Return_to_owner","Transfer"])

	from sklearn.metrics import classification_report, accuracy_score, log_loss
	print(accuracy_score(y_val, y_pred))
	print(log_loss(y_val, y_probs))
	print(log.classes_)
	#print(log.feature_importances_)

#log-loss in training data corresponds to test data (result almost similar)

#########################################################################################################
#### Model Fitting
#########################################################################################################
#fit all data
log.fit(model_train, train_outcome)

y_probs = log.predict_proba(model_test)
#print(y_probs)

results = pd.read_csv("sample_submission.csv")
#print(results)

#each result has their corresponding probabilistic value
results["Adoption"] = y_probs[:,0]
results["Died"] = y_probs[:,1]
results["Euthanasia"] = y_probs[:,2]
results["Return_to_owner"] = y_probs[:,3]
results["Transfer"] = y_probs[:,4]

results.to_csv("weight_aggressive_allergic.csv",index = False)

#########################################################################################################
#### Plotting Result
#########################################################################################################

import seaborn as sns 
import matplotlib.pyplot as plt 

#plot importance
plt.figure(0)
plt.title("Feature Importance")
print(log.feature_importances_)
print(shelter_train.columns)
importance = log.feature_importances_
sns.barplot(y=shelter_train.columns, x=importance)

# plot shelter train correlation
axis = {
		"AnimalType",
		"AgeuponOutcome",
		"Breed",
		"Color",
		"Year",
		"Month",
		"Day",
		"Hour",
		"Minute",
		"Virginity",
		"Sex",
		"has_name",
		"Aggresiveness",
		"Allergic",
		"Weight" 
		}
count = 1
for ax in axis:
	x = shelter_train[ax]
	y = train_outcome
	plt.figure(count)
	plt.title(ax)
	sns.countplot(x=x, hue=y)
	count = count + 1

plt.show()

#########################################################################################################
#### Reference
#########################################################################################################
# get_dummies
# df
#    A  B  C
# 0  a  c  1
# 1  b  c  2
# 2  a  b  3

# pd.get_dummies(df)
#    C  A_a  A_b  B_b  B_c
# 0  1    1    0    0    1
# 1  2    0    1    0    1
# 2  3    1    0    1    0