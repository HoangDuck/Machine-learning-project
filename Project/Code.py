'''
The following code is mainly from Chap 2, Géron 2019 
See https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb

LAST REVIEW: Oct 2020
'''

# In[0]: IMPORTS 
from math import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# np.set_printoptions(threshold = np.inf)
# pd.options.display.max_columns = 20
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
#from statistics import mean


# In[1]: LOOK AT THE BIG PICTURE (DONE)
# Dự đoán lương của một người hoặc là làm một 
# phần mềm, công cụ để dự đoán lương khi mik apply 
# và mik sử dụng lương deal cho hợp lý

# In[2]: GET THE DATA (DONE). LOAD DATA
raw_data = pd.read_csv('Raw_DataSet/IT Salary Survey EU  2020.csv')


# In[3]: DISCOVER THE DATA TO GAIN INSIGHTS
# 3.1 Quick view of the data
print('\n____________________________________ Dataset info ____________________________________')
print(raw_data.info())              
print('\n____________________________________ Some first data examples ____________________________________')
print(raw_data.head(6)) 
print('\n____________________________________ Counts on a feature ____________________________________')
print(raw_data['Your main technology / programming language'].value_counts()) 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(raw_data.describe())    
print('\n____________________________________ Get specific rows and cols ____________________________________')     
print(raw_data.iloc[[0,5,20], [2, 7]] ) # Refer using column ID
#%%Plot
# # 3.2 Scatter plot b/w 2 features
# if 1:
#     raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="SỐ PHÒNG", alpha=0.2)
#     plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
#     plt.show()      
# if 1:
#     raw_data.plot(kind="scatter", y="GIÁ - TRIỆU ĐỒNG", x="DIỆN TÍCH - M2", alpha=0.2)
#     plt.show()

# # 3.3 Scatter plot b/w every pair of features
# if 1:
#     from pandas.plotting import scatter_matrix   
#     features_to_plot = ["GIÁ - TRIỆU ĐỒNG", "SỐ PHÒNG", "SỐ TOILETS", "DIỆN TÍCH - M2"]
#     scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
#     plt.savefig('figures/scatter_mat_all_feat.png', format='png', dpi=300)
#     plt.show()

# # 3.4 Plot histogram of 1 feature
# if 1:
#     from pandas.plotting import scatter_matrix   
#     features_to_plot = ["SỐ PHÒNG"]
#     scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
#     plt.show()

# # 3.5 Plot histogram of numeric features
# if 1:
#     #raw_data.hist(bins=10, figsize=(10,5)) #bins: no. of intervals
#     raw_data.hist(figsize=(10,5)) #bins: no. of intervals
#     plt.rcParams['xtick.labelsize'] = 10
#     plt.rcParams['ytick.labelsize'] = 10
#     plt.tight_layout()
#     plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
#     plt.show()

#%% 3.6 Compute correlations b/w features
corr_matrix = raw_data.corr()
print(corr_matrix) # print correlation matrix
#%% 3.7 Try add features and classify data
#Function for classifying data:
#%%3.7.1 Execute adding features and classify data programming languages
import json
#Global variables
data=None
file_path='Json_Files/ListofPL.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def CheckAlphabet(text_check):
    if((text_check >= 'a' and text_check <= 'z') or (text_check >= 'A' and text_check <= 'Z')):
        return True
    else:
        return False
def CLanguagesChecking(text_raw_data,key="c"):
    if(text_raw_data.find("Embedded C")!=-1):
        return True
    index=text_raw_data.find(key)
    if(index==-1):
        return False
    index_increased=index+1
    index_decreased=index-1
    try:
        if(text_raw_data[index_increased]==","or text_raw_data[index_increased]=="/"):
            return True
        elif text_raw_data[index_increased]=="#" or text_raw_data[index_increased]=="+":
            return False
        elif(text_raw_data[index_decreased]=="/" and (text_raw_data[index_increased]!="#" and text_raw_data[index_increased]!="+" )):
            if((CheckAlphabet(text_raw_data[index_increased]) and CheckAlphabet(text_raw_data[index_decreased]))==False):
                return True
            return False
        elif((CheckAlphabet(text_raw_data[index_increased]) and CheckAlphabet(text_raw_data[index_decreased]))==False):
            return True
    except:
        if(text_raw_data[index]=="c"or text_raw_data[index]=="C"):
            return True
    return False
def ClassifyData(data,text_raw_data):
    result=""
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            if(key=="C" or key=="c" or key=="R"):
                if(text_raw_data=="C" or text_raw_data=="c"):
                    result+=data[key]+"; "
                elif(text_raw_data=="R"):
                    result+=data[key]+"; "
                elif(CLanguagesChecking(text_raw_data,"c")and key=="c"):
                    result+=data[key]+"; "
                elif(CLanguagesChecking(text_raw_data,"C")):
                    result+=data[key]+"; "
            else:
                result+=data[key]+"; "
    return result
#%%Executing
n=raw_data.index
num=len(n)
list_Programming_Languages=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Programming_Languages.append('No info')
    else:
        p=ClassifyData(data,s)#Them ham tra ve ngon ngu
        list_Programming_Languages.append(p)
raw_data["Programming languages"]=list_Programming_Languages
#%%Export files
import json
with open('ListTest.json', 'w') as f:
    json.dump(list_Programming_Languages, f)
#%%3.7.2 Execute adding features and classify frameworks or libraries
import json
#Global variables
data=None
file_path='Json_Files/ListFM.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%%Executing
raw_data["Frameworks / Libs"]=ClassifyData(data,raw_data["Your main technology / programming language"])#Them ham tra ve frameworks / libs
#%%3.7.3 Execute adding features and classify databases
import json
#Global variables
data=None
file_path='Json_Files/ListDB.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%%Executing
raw_data["Databases"]=ClassifyData(data,raw_data["Your main technology / programming language"])#Them ham tra ve databases
#%%3.7.4 Execute adding features and classify designs
import json
#Global variables
data=None
file_path='Json_Files/ListDesign.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%%Executing
raw_data["Design"]=ClassifyData(data,raw_data["Your main technology / programming language"])#Them ham tra ve design
#%%3.7.5 Execute adding features and classify clouds
import json
#Global variables
data=None
file_path='Json_Files/ListCloud.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%%Executing
raw_data["Clouds"]=raw_data["Your main technology / programming language"]#Them ham tra ve clouds
#%%3.7.6 Execute adding features and classify platforms
import json
#Global variables
data=None
file_path='Json_Files/ListPlatform.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            if(key=="Linux Kernel"):
                continue
            result+=data[key]+"; "
    return result
#%%Executing
raw_data["Platform"]=raw_data["Your main technology / programming language"]#Them ham tra ve platforms
#%%3.7.7 Execute adding features and classify devops-tools
import json
#Global variables
data=None
file_path='Json_Files/ListFM.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%%Executing
raw_data["DevOps tools"]=raw_data["Your main technology / programming language"]#Them ham tra ve devops tools
# In[04]: PREPARE THE DATA 
# 4.1 Remove unused features
raw_data.drop(columns = ["Timestamp", "Age", "Gender", "City", 
                         "Years of experience in Germany", "Other technologies/programming languages you use often",
                         "Yearly bonus + stocks in EUR", "Annual brutto salary (without bonus and stocks) one year ago. Only answer if staying in the same country",
                         "Annual bonus+stocks one year ago. Only answer if staying in same country","Number of vacation days",
                         "Main language at work","Have you lost your job due to the coronavirus outbreak?",
                         "Have you been forced to have a shorter working week (Kurzarbeit)? If yes, how many hours per week"], inplace=True) 
#Store dataset

#%% 4.2 Split training-test set and NEVER touch test set until test phase
method = 2
if method == 1: # Method 1: Randomly select 20% of data for test set. Used when data set is large
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42) # set random_state to get the same training set all the time, 
                                                                                     # otherwise, when repeating training many times, your model may see all the data
elif method == 2: # Method 2: Stratified sampling, to remain distributions of important features, see (Geron, 2019) page 56
    # Create new feature "KHOẢNG GIÁ": the distribution we want to remain
    raw_data["KHOẢNG GIÁ"] = pd.cut(raw_data["GIÁ - TRIỆU ĐỒNG"],
                                    bins=[0, 2000, 4000, 6000, 8000, np.inf],
                                    labels=[2,4,6,8,100]) # use numeric labels to plot histogram
    
    # Create training and test set
    from sklearn.model_selection import StratifiedShuffleSplit  
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
    for train_index, test_index in splitter.split(raw_data, raw_data["KHOẢNG GIÁ"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]      
    
    # See if it worked as expected
    if 1:
        raw_data["KHOẢNG GIÁ"].hist(bins=6, figsize=(5,5)); #plt.show();
        test_set["KHOẢNG GIÁ"].hist(bins=6, figsize=(5,5)); plt.show()

    # Remove the new feature
    print(train_set.info())
    for _set_ in (train_set, test_set):
        _set_.drop(columns="KHOẢNG GIÁ", inplace=True) 
    print(train_set.info())
    print(test_set.info())
print('\n____________________________________ Split training an test set ____________________________________')     
print(len(train_set), "train +", len(test_set), "test examples")
print(train_set.head(4))

#%% 4.3 Separate labels from data, since we do not process label values
train_set_labels = train_set["Yearly brutto salary (without bonus and stocks) in EUR"].copy()
train_set = train_set.drop(columns = "Yearly brutto salary (without bonus and stocks) in EUR") 
test_set_labels = test_set["Yearly brutto salary (without bonus and stocks) in EUR"].copy()
test_set = test_set.drop(columns = "Yearly brutto salary (without bonus and stocks) in EUR") 

#%% 4.4 Define pipelines for processing data. 
# INFO: Pipeline is a sequence of transformers (see Geron 2019, page 73). For step-by-step manipulation, see Details_toPipeline.py 

# 4.4.1 Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         

num_feat_names = ['DIỆN TÍCH - M2', 'SỐ PHÒNG', 'SỐ TOILETS'] # =list(train_set.select_dtypes(include=[np.number]))
cat_feat_names = ['QUẬN HUYỆN', 'HƯỚNG', 'GIẤY TỜ PHÁP LÝ'] # =list(train_set.select_dtypes(exclude=[np.number])) 

# 4.4.2 Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
    ])    

# INFO: Try the code below to understand how a pipeline works
if 10:
    trans_feat_values_1 = cat_pipeline.fit_transform(train_set)

    # The above line of code is equavalent to the following code:     
    selector  = ColumnSelector(cat_feat_names)
    temp_feat_values = selector.fit_transform(train_set) 
    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)
    temp_feat_values = imputer.fit_transform(temp_feat_values) 
    one_hot_encoder = OneHotEncoder()
    trans_feat_values_2 = one_hot_encoder.fit_transform(temp_feat_values)
    if 0: 
        # See the encoded features
        print(one_hot_encoder.categories_) # INFO: categories_ is an array of array: categories_[0] is the array of feature 1, categories_[1] is the array of feature 2,...
        # NOTE: OneHotEncoder turns 1 features into N features, where N is the no. of values in that feature
        # e.g., feature "HƯỚNG" having 5 values 'Đông', 'Tây', 'Nam', 'Bắc', 'NO INFO', will become 5 features corresponding with its values 
        print(one_hot_encoder.get_feature_names(cat_feat_names))
        print("No. of one-hot columns: " + str(one_hot_encoder.get_feature_names(cat_feat_names).shape[0]))
        print(trans_feat_values_2[[0,1,2],:].toarray()) # toarray() convert sparse to dense array
    
    # Check if trans_feat_values_1 and trans_feat_values_2 are the same
    #print(trans_feat_values_1.toarray() == trans_feat_values_2.toarray())
    print(np.array_equal(trans_feat_values_1.toarray(), trans_feat_values_2.toarray()))

# 4.4.3 Define MyFeatureAdder: a transformer for adding features "TỔNG SỐ PHÒNG",...  
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_TONG_SO_PHONG = True): # MUST NO *args or **kargs
        self.add_TONG_SO_PHONG = add_TONG_SO_PHONG
    def fit(self, feature_values, labels = None):
        return self  # nothing to do here
    def transform(self, feature_values, labels = None):
        SO_PHONG_id, SO_TOILETS_id = 1, 2 # column indices in num_feat_names. can't use column names b/c the transformer SimpleImputer removed them
        # NOTE: a transformer in a pipeline ALWAYS return dataframe.values (ie., NO header and row index)
        
        TONG_SO_PHONG = feature_values[:, SO_PHONG_id] + feature_values[:, SO_TOILETS_id]
        if self.add_TONG_SO_PHONG:
            feature_values = np.c_[feature_values, TONG_SO_PHONG] #concatenate np arrays
        return feature_values

# 4.4.4 Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), # copy=False: imputation will be done in-place 
    ('attribs_adder', MyFeatureAdder(add_TONG_SO_PHONG = True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
    ])  
#feature scaling biến đổi khoảng giá trị các thuộc tính bằng nhau như thay vì 1-500 thì đưa về 0-1 hoặc -1 - 1  
# 4.4.5 Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# 4.5 Run the pipeline to process training data           
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________________________________ Processed feature values ____________________________________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 32 cols of onehotvector for categorical features.' %(len(num_feat_names)))

# (optional) Add header to create dataframe. Just to see. We don't need header to run algorithms 
if 0: 
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    columns_header = train_set.columns.tolist() + ["TỔNG SỐ PHÒNG"] + onehot_cols
    for name in cat_feat_names:
        columns_header.remove(name)
    processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns = columns_header)
    print('\n____________________________________ Processed dataframe ____________________________________')
    print(processed_train_set.info())
    print(processed_train_set.head())



# %%
