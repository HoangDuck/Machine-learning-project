#%%Test
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
                result+=data[key]+"; "
    return result
#%%Run
result=ClassifyData(data,'Swift, Objective-C, iOS')
print(result)

# %%
