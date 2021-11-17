#%%Test
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
#%%Run
result=ClassifyData(data,'AWS,android Linux Kernel Linux iOSGCP, Python,K8s')
print(result)

# %%
