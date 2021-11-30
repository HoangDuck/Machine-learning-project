'''
The following code is mainly from Chap 2, Géron 2019 
See https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb

LAST REVIEW: Oct 2020
'''
'''

Ten de tai:

'''
# In[0]: CÀI ĐẶT THƯ VIỆN
from math import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline #Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean 

# In[1]: Tổng quan về đề tài:
# Dự đoán lương của một người hoặc là làm một 
# phần mềm, công cụ để dự đoán lương khi mik apply 
# và mik sử dụng lương deal cho hợp lý

# In[2]: LẤY DỮ LIỆU
'''
    - Dữ liệu được lấy từ Kaggle
    Nguồn: https://www.kaggle.com/parulpandey/2020-it-salary-survey-for-eu-region
    - Tập dữ liệu chứa những thông tin khảo sát mức lương của những nhân lực làm việc cho mảng IT ở vùng châu Âu.
    - Tập dữ liệu nhóm em sử dụng là các thông tin khảo sát của năm 2020.
    - Tác giả: Parul Pandey
    - Lần cập nhật cuối của tập dữ liệu là 10 tháng trước.
'''
raw_data = pd.read_csv('Raw_DataSet/IT Salary Survey EU  2020.csv')

# In[3]: KHÁM PHÁ DỮ LIỆU:
# 3.1 Quan sát tập dữ liệu
print('\n____________________________________ Dataset info ____________________________________')
print(raw_data.info())              
print('\n____________________________________ Some first data examples ____________________________________')
print(raw_data.head(6)) 
print('\n____________________________________ Counts on a feature ____________________________________')
print(raw_data['Your main technology / programming language'].value_counts()) 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(raw_data.describe())    
print('\n____________________________________ Get specific rows and cols ____________________________________')     
print(raw_data.iloc[[0,5,20], [7, 8]] ) # Refer using column ID
#%% Vẽ biểu đồ
# 3.2. Biểu đồ scatter plot giữa độ tuổi và lương của nhân lực IT ở châu Âu năm 2020
if 1:
    raw_data.plot(kind="scatter", y="Yearly brutto salary (without bonus and stocks) in thoundsands EUR", x="Age", alpha=0.2)
    plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()      

# 3.3. Biểu đồ histogram thống kê giới tính nhân lực IT ở châu Âu năm 2020:
if 1:
    data = raw_data["Gender"]
    plt.hist(data,color = 'r')
# 3.4. Biểu đồ histogram về tuổi của nhân lực làm IT ở châu Âu năm 2020:
if 1:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["Age"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.show()

# 3.5 Biểu đồ histogram về lương của nhân lực IT (Chưa tính thêm tiền thưởng) ở châu Âu năm 2020:
if 1:
    data = raw_data["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"]
    plt.hist(data,color = 'r')

#%% 3.6 Tính toán độ tương quan giữa các thuộc tính
corr_matrix = raw_data.corr()
print(corr_matrix) 
#%% Lọc dữ liệu
# Loại bỏ các sample có quá nhiều giá trị null và các giá trị outliers
# Chuẩn hóa lại các dữ liệu (Sửa lỗi chính tả, lỗi phông và viết hoa thường đồng đều)
#%% 3.7 Thêm các thuộc tính mới
#Function for classifying data:
#%%3.7.1 Thực hiện thêm cột dữ liệu về ngôn ngữ lập trình 
# và thêm dữ liệu ngôn ngữ lập trình và cột đó:
# Cài thư viện json để đọc file json dưới dạng dữ liệu dictionaries
import json
# Khai báo biến data chứa dữ liệu dạng dictionaries đọc từ file Json
data=None
file_path='Json_Files/ListofPL.json'
# Mở file Json
with open(file_path) as json_file:
    data = json.load(json_file)
# Kiểm tra chữ đó thuộc bảng chữ cái
def CheckAlphabet(text_check):
    if((text_check >= 'a' and text_check <= 'z') or (text_check >= 'A' and text_check <= 'Z')):
        return True
    else:
        return False
# Kiểm tra ngôn ngữ đó là ngôn ngữ C
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
            if(text_raw_data[index_decreased]=="-"):
                return False
            return True
        elif text_raw_data[index_increased]=="#" or text_raw_data[index_increased]=="+":
            return False
        elif(text_raw_data[index_decreased]=="/" and (text_raw_data[index_increased]!="#" and text_raw_data[index_increased]!="+" )):
            if((CheckAlphabet(text_raw_data[index_increased]) and CheckAlphabet(text_raw_data[index_decreased]))==False):
                return True
            return False
        elif(CheckAlphabet(text_raw_data[index_increased])):
            return False
        elif((CheckAlphabet(text_raw_data[index_increased]) and CheckAlphabet(text_raw_data[index_decreased]))==False):
            return True
    except:
        if((text_raw_data[index]=="c"or text_raw_data[index]=="C")and text_raw_data[index-1]!="-"):
            return True
        else:
            return False
    return False
# Phân loại dữ liệu
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            if(key=="C" or key=="c" or key=="R"):
                if(text_raw_data=="C" or text_raw_data=="c"):
                    result+=data[key]+"; "
                elif(key=="R"):
                    if text_raw_data=="R":
                        result+=data[key]+"; "
                    elif CheckAlphabet(text_raw_data[index+1])==False:
                        result+=data[key]+"; "
                elif(CLanguagesChecking(text_raw_data,"c")and key=="c"):
                    result+=data[key]+"; "
                elif(CLanguagesChecking(text_raw_data,"C")and key=="C"):
                    result+=data[key]+"; "
            else:
                if(key=="go"):
                    index_find_Django=text_raw_data.find("Django")
                    if(index!=-1):
                        continue
                if(key=="Java" or key=="java"):
                    try:
                        if(text_raw_data[index+4]=="s"or text_raw_data[index+4]=="S"):
                            continue
                    except:
                        result+=data[key]+"; "
                        continue
                if(key=="Js"and text_raw_data=="Js, reactJS "):
                    continue
                result+=data[key]+"; "
    return result
#%%Thực thi
n=raw_data.index
num=len(n)
# Biến chứa dữ liệu ngôn ngữ lập trình sau khi được xử lý 
# và chuẩn bị chèn vào dataframe
list_Programming_Languages=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Programming_Languages.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Programming_Languages.append(p)
# Dữ liệu sau khi được xử lý và chèn vào dataframe
raw_data["Programming languages"]=list_Programming_Languages

#%%3.7.2 Thêm cột dữ liệu và phân loại các frameworks hoặc libraries
import json
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
#%%Thực thi
list_Frameworks_Libraries=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Frameworks_Libraries.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Frameworks_Libraries.append(p)
raw_data["Frameworks / Libs"]=list_Frameworks_Libraries
#%%3.7.3 Thêm cột dữ liệu và phân loại dữ liệu của databases
import json
data=None
file_path='Json_Files/ListDB.json'
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
#%%Thực thi
list_Databases=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Databases.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Databases.append(p)
raw_data["Databases"]=list_Databases
#%%3.7.4 Thêm cột dữ liệu và phân loại dữ liệu về công cụ, mô hình thiết kế
import json
data=None
file_path='Json_Files/ListDesign.json'
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
#%%Thực thi
list_Design=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Design.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Design.append(p)
raw_data["Design"]=list_Design
#%%3.7.5 Thêm dữ liệu và phân loại dữ liệu về Cloud
import json
data=None
file_path='Json_Files/ListCloud.json'
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
#%%Thực thi
list_CLouds=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_CLouds.append('No info')
    else:
        p=ClassifyData(data,s)
        list_CLouds.append(p)
raw_data["Clouds"]=list_CLouds
#%%3.7.6 Platforms
import json
data=None
file_path='Json_Files/ListPlatform.json'
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
#%%Thực thi
list_Platforms=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Platforms.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Platforms.append(p)
raw_data["Platform"]=list_Platforms
#%%3.7.7 Thêm dữ liệu cột dữ liệu của các công cụ DevOps
import json
data=None
file_path='Json_Files/ListDevOps-Tools.json'
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
#%%Thực thi
list_DevOps_Tools=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_DevOps_Tools.append('No info')
    else:
        p=ClassifyData(data,s)
        list_DevOps_Tools.append(p)
raw_data["DevOps tools"]=list_DevOps_Tools
#%%3.7.8. Điều chỉnh dữ liệu tổng số năm kinh nghiệm
import json
data=None
file_path='Json_Files/YearOfExperience.json'
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%%Thực thi
list_Year_Experience=[]
for x in range(num):
    try:
        s=raw_data["Total years of experience"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Year_Experience.append(s)
            continue
        else:
            list_Year_Experience.append(result)
    except:
        list_Year_Experience.append(raw_data["Total years of experience"][x])
        continue
raw_data["Total years of experience"]=list_Year_Experience
#%%3.7.9. Điều chỉnh dữ liệu các loại hợp đồng
import json
data=None
file_path='Json_Files/TypeContract.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%%Thực thi
list_Contract=[]
for x in range(num):
    try:
        s=raw_data["Contract duration"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Contract.append(s)
            continue
        else:
            list_Contract.append(result)
    except:
        list_Contract.append(raw_data["Contract duration"][x])
        continue
raw_data["Contract duration"]=list_Contract
#%%3.7.10. Điều chỉnh dữ liệu quy mô công ty
import json
data=None
file_path='Json_Files/CompanySize.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%%Thực thi
list_Company_Size=[]
for x in range(num):
    try:
        s=raw_data["Company size"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Company_Size.append(s)
            continue
        else:
            list_Company_Size.append(result)
    except:
        list_Company_Size.append(raw_data["Company size"][x])
        continue
raw_data["Company size"]=list_Company_Size
#%%3.7.11. Điều chỉnh dữ liệu kiểu công ty
import json
data=None
file_path='Json_Files/Company_type.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%%Thực thi
list_Company_Type=[]
for x in range(num):
    try:
        s=raw_data["Company type"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Company_Type.append(s)
            continue
        else:
            list_Company_Type.append(result)
    except:
        list_Company_Type.append(raw_data["Company type"][x])
        continue
raw_data["Company type"]=list_Company_Type
#%%3.7.12. Điều chỉnh dữ liệu tiền hỗ trợ, trợ cấp trong năm 2020
import json
data=None
file_path='Json_Files/Additional_monetary_support.json'
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%%Thực thi
list_Additional_Support=[]
for x in range(num):
    try:
        s=raw_data["Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Additional_Support.append(s)
            continue
        else:
            list_Additional_Support.append(result)
    except:
        list_Additional_Support.append(raw_data["Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR"][x])
        continue
raw_data["Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR"]=list_Additional_Support

# In[04]: PREPARE THE DATA
# 4.1 Loại bỏ những cột không sử dụng
raw_data.drop(columns = ["Timestamp", "Age", "Gender", "City", 
                         "Years of experience in Germany", "Other technologies/programming languages you use often",
                         "Yearly bonus + stocks in EUR", "Annual brutto salary (without bonus and stocks) one year ago. Only answer if staying in the same country",
                         "Annual bonus+stocks one year ago. Only answer if staying in same country","Number of vacation days",
                         "Main language at work","Have you lost your job due to the coronavirus outbreak?",
                         "Have you been forced to have a shorter working week (Kurzarbeit)? If yes, how many hours per week",
                         "Your main technology / programming language"], inplace=True) 
#%%Lưu tập dữ liệu qua bước tiền xử lý
raw_data.to_csv(r'C:\Users\ADMIN\Máy tính\AI\Machine_Learning\Final_Project\DataSet-FinalProject (Processing)\Project\DataSet_Filtered\export_dataset.csv', index = False, header=True)
#%% 4.2 Tách tập dữ liệu trên thành tập train và tập test
split = True
if split: # Stratified sampling
    '''
        Cách tạo test set này thì lấy dữ liệu mức lương trong tập dữ liệu đưa về thành các khoản chung
        Và sau đó lấy mỗi khoảng đã chia một ít dữ liệu 20%, mỗi tập dữ liệu nhỏ đại diện tốt cho khoảng đó.
        Khi tập hợp các khoảng dữ liệu nhỏ ấy thì ta có tập test_set mà vẫn đảm bảo nó có tính ngẫu nhiên
        và đại diện được cho cả tập dữ liệu.
    '''
    # Tạo ra thuộc tính mới "Salary_About"
    raw_data["Salary_About"] = pd.cut(raw_data["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"],
                                    bins=[0, 100, 200, 300, 400, np.inf],
                                    labels=[100,200,300,400,500]) # use numeric labels to plot histogram
    
    # Tạo tập train và tập test; Tập test để đến bước cuối cùng
    from sklearn.model_selection import StratifiedShuffleSplit  
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
    for train_index, test_index in splitter.split(raw_data, raw_data["Salary_About"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]      
    
    # Vẽ biểu đồ lấy mẫu ở tập dữ liệu
    if 0:
        raw_data["Salary_About"].hist(bins=6, figsize=(5,5)); #plt.show();
        test_set["Salary_About"].hist(bins=6, figsize=(5,5)); plt.show()

    #Loại bỏ thuộc tính salary_about
    for _set_ in (train_set, test_set):
        _set_.drop(columns="Salary_About", inplace=True) 
    print(train_set.info())
    print(test_set.info())
print('\n____________________________________ Split training an test set ____________________________________')     
print(len(train_set), "train +", len(test_set), "test examples")
#print(train_set.head(4))

#%% 4.3 Tiến hành tách cột label ở tập train và tập test và không xử lý cột label
# Ở trong đồ án này thì label - output bài toán - là:
# cột mức lương được tính bằng ngàn EUR.
train_set_labels = train_set["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"].copy()
train_set = train_set.drop(columns = "Yearly brutto salary (without bonus and stocks) in thoundsands EUR") 
test_set_labels = test_set["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"].copy()
test_set = test_set.drop(columns = "Yearly brutto salary (without bonus and stocks) in thoundsands EUR") 

#%% 4.4 Cài đặt các Pipeline để tiến hành xử lý dữ liệu
# INFO: Pipeline is a sequence of transformers (see Geron 2019, page 73). For step-by-step manipulation, see Details_toPipeline.py 

# 4.4.1 Định nghĩa hàm chọn cột
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         
# Lưu tập list chứa cột dữ liệu dạng số:
num_feat_names = ['Total years of experience', 
'Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR']
# Lưu tập list chứa cột dữ liệu dạng chữ (Phân loại):
cat_feat_names = ['Position ', 'Seniority level', 'Employment status',
'Contract duration','Company size','Company type','Programming languages','Frameworks / Libs',
'Databases','Design','Clouds','Platform','DevOps tools']

# 4.4.2 Pipeline xử lý dữ liệu dạng chữ (phân loại):
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    #Điền chữ "No info" khi phát hiện chỗ dữ liệu chữ còn trống
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "No info", copy=True)),
     # chuyển đổi dữ liệu chữ qua one-hot vectors
    ('cat_encoder', OneHotEncoder())
    ])    

# 4.4.4 Pipeline xử lý dữ liệu dạng số:
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    # Điền vào chỗ dữ liệu số còn trống bằng giá trị trung vị
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),
    # Tiến hành fearture scaling để đưa các giá trị trong tập dữ liệu về các khoảng bằng nhau
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))
    ])  
#feature scaling biến đổi khoảng giá trị các thuộc tính bằng nhau như thay vì 1-500 thì đưa về 0-1 hoặc -1 - 1  
# 4.4.5 Kết hợp hai pipeline trên với nhau
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# 4.5 Tiến hành chạy pipeline xử lý chuyển đổi tập dữ liệu        
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________________________________ Processed feature values ____________________________________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 32 cols of onehotvector for categorical features.' %(len(num_feat_names)))

# In[5]: TRAINING MODELS VÀ VALIDATION TEST  
'''
    Ở bước này tiến hành chạy các model, đo độ chính xác và độ lệch ở mỗi model
    Tiến hành chạy validation đảm bảo chọn model khách quan
    Sau khi chọn model thì tiến hành bước fine-tune
    Ở bước này những model được thực hiện:
    + Linear Regression
    + Decision tree
    + Random Forest
    + Polinomial
    + SVM
'''
# 5.1 LinearRegression model
# 5.1.1 Training: Học ra một linear regression hypothesis sử dụng training data 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(processed_train_set_val, train_set_labels)
print('\n____________________________________ LinearRegression ____________________________________')
print('Learned parameters: ', model.coef_)

# 5.1.2 Tính R2 score và root mean squared error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
        
#%% 5.1.3 Predict labels for some training instances
print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

#%% 5.1.4 Lưu trữ các models
import joblib # new lib
def store_model(model, model_name = ""):
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'saved_objects/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('saved_objects/' + model_name + '_model.pkl')
    return model
store_model(model)


#%% 5.2 Try DecisionTreeRegressor model
# Training
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ DecisionTreeRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)
# Predict labels for some training instances
print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.3 Try RandomForestRegressor model
# Training (NOTE: may take time if train_set is large)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5, random_state=42) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ RandomForestRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.4 Try polinomial regression model
# NOTE: polinomial regression can be treated as (multivariate) linear regression where high-degree features x1^2, x2^2, x1*x2... are seen as new features x3, x4, x5... 
# hence, to do polinomial regression, we add high-degree features to the data, then call linear regression
# 5.5.1 Training. NOTE: may take a while 
from sklearn.preprocessing import PolynomialFeatures
poly_feat_adder = PolynomialFeatures(degree = 2) # add high-degree features to the data
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name = "PolinomialRegression")      
else:
    model = load_model("PolinomialRegression")
# 5.4.2 Compute R2 score and root mean squared error
print('\n____________________________________ Polinomial regression ____________________________________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 5.4.3 Predict labels for some training instances
print("Predictions: ", model.predict(train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.5 Evaluate with K-fold cross validation 
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
#from sklearn.model_selection import cross_val_predict

#cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
#cv2 = StratifiedKFold(n_splits=10, random_state=42); 
#cv3 = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
print('\n____________________________________ K-fold cross validation ____________________________________')

run_evaluation = 1
if run_evaluation:
    # Evaluate LinearRegression
    model_name = "LinearRegression" 
    model = LinearRegression()             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))

    # Evaluate DecisionTreeRegressor
    model_name = "DecisionTreeRegressor" 
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))

    # Evaluate RandomForestRegressor
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    
    # Evaluate Polinomial regression
    model_name = "PolinomialRegression" 
    model = LinearRegression()
    nmse_scores = cross_val_score(model, train_set_poly_added, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "PolinomialRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')


# In[6]: FINE-TUNE CÁC MODELS 
# INFO: find best hyperparams (param set before learning, e.g., degree of polynomial in poly reg, no. of trees in rand forest, no. of layers in neural net)
# Here we fine-tune RandomForestRegressor and PolinomialRegression
print('\n____________________________________ Fine-tune models ____________________________________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    print('Best estimator: ', grid_search.best_estimator_)  
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

method = 1
# 6.1 Method 1: Grid search (try all combinations of hyperparams in param_grid)
if method == 1:
    from sklearn.model_selection import GridSearchCV
    
    run_new_search = 1      
    if run_new_search:
        # 6.1.1 Fine-tune RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]} ]
            # Train across 5 folds, hence a total of (12+6)*5=90 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")      

        # 6.1.2 Fine-tune Linear regression          
        # model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()), # add high-degree features
        #                    ('lin_reg', LinearRegression()) ]) 
        # param_grid = [
        #     # try 3 values of degree
        #     {'poly_feat_adder__degree': [1, 2, 3]} ] # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
        #     # Train across 5 folds, hence a total of 3*5=15 rounds of training 
        # grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        # grid_search.fit(processed_train_set_val, train_set_labels)
        # joblib.dump(grid_search,'saved_objects/PolinomialRegression_gridsearch.pkl') 
        # print_search_result(grid_search, model_name = "PolinomialRegression") 
    else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        grid_search = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name = "PolinomialRegression") 


# In[7]: ANALYZE AND TEST YOUR SOLUTION
# NOTE: solution is the best model from the previous steps. 

# 7.1 Pick the best model - the SOLUTION
# Pick Random forest
search = joblib.load('saved_objects/RandomForestRegressor_randsearch.pkl')
best_model = search.best_estimator_
# Pick Linear regression
#best_model = joblib.load('saved_objects/LinearRegression_model.pkl')

print('\n____________________________________ ANALYZE AND TEST YOUR SOLUTION ____________________________________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUION")   

# 7.2 Analyse the SOLUTION to get more insights about the data
# NOTE: ONLY for rand forest
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + ["TỔNG SỐ PHÒNG"] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

# 7.3 Run on test data
processed_test_set = full_pipeline.transform(test_set)  
# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("Predictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')


'''
Kết quả hiện chưa tốt. 
Gợi ý cả thiện:
1. Xóa các samples bất thường (giá quá cao, quá thấp)
2. Xóa cột HƯỚNG, hoặc tốt nhất là thêm dữ liệu (2000+)
'''


# In[8]: LAUNCH, MONITOR, AND MAINTAIN YOUR SYSTEM
# Hướng phát triển