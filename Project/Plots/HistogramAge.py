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
#%%
data = raw_data["Gender"]
plt.hist(data,color = 'r')


# %%
if 1:
    raw_data.plot(kind="scatter", y="Yearly brutto salary (without bonus and stocks) in thoundsands EUR", x="Age", alpha=0.2)
    plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()   