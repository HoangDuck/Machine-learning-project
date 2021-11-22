#%%
import json
data=None
file_path='Json_Files/YearOfExperience.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
print(data)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%%
p=ClassifyData(data,'-')
print(p)

# %%
a="l"
print(type(a))
# %%
import numpy as np
list_Year_Experience=np.zeros


# %%
