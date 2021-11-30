#%%Classify company type
import json
data=None
file_path='Json_Files/ListPositions.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    if(text_raw_data=="Analyst" or text_raw_data=="Marketing Analyst" or text_raw_data=="Product Analyst"):
        result='Analytics engineer'
        return result
    elif(text_raw_data=="Consultant" or text_raw_data=="Application Consultant"
     or text_raw_data=="BI Consultant" or text_raw_data=="BI IT Consultant"
     or text_raw_data=="ERP Consultant" or text_raw_data=="SAP BW Senior Consultant" 
     or text_raw_data=="SAP Consultant"):
        result='Consultant'
        return result
    elif(text_raw_data=="Architect" or text_raw_data=="Data Architect"):
        result='Architect'
        return result
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%%
p=ClassifyData(data,'Software Architect')
print(p)
# %%

# %%
