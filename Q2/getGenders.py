import pandas as pd
import requests, json
from time import sleep
"""
@Author : https://github.com/block8437/gender.py/blob/master/gender.py
"""
def getGenders(names):
    url = ""
    cnt = 0
    if not isinstance(names,list):
        names = [names,]

    for name in names:
        if url == "":
            url = "name[0]=" + name
        else:
            cnt += 1
            url = url + "&name[" + str(cnt) + "]=" + name

    print("https://api.genderize.io?" + url)
    req = requests.get("https://api.genderize.io?" + url)
    print(req)
    results = json.loads(req.text)
    print(results)
    retrn = []
    for result in results:
        if result["gender"] is not None:
            retrn.append((result["gender"], result["probability"], result["count"]))
        else:
            retrn.append((u'None',u'0.0',0.0))
    return retrn

def get_frame(txt_path, label=0):
    accept17 = pd.read_table(txt_path, header=None)
    accept17.columns = ['title', 'authors']
    accept17['len_authors'] = accept17['authors'].str.len()
    accept17['len_title'] = accept17['title'].str.len()
    accept17['num_words'] = accept17['title'].str.split(" ").str.len()
    accept17['num_authors'] = accept17['authors'].str.split(",").str.len()
    accept17['accepted'] = label
    first_names = accept17['authors'].str.split(",").apply(lambda x : x[0].split(" ")[0]).tolist()
    res_first_names = []
    genders2 = []
    probabilties2 = []
    counts2 = []
    for i in range(0, len(first_names), 10):
        output = getGenders(first_names[i:min(i+10, len(first_names))])
        print(output)
        sleep(1)
        for out in output:
            genders2.append(out[0])
            probabilties2.append(out[1])
            counts2.append(out[2])
        res_first_names += output
    accept17['gender'] = genders2
    accept17['prob'] = probabilties2
    accept17['count'] = counts2
    accept17 = pd.concat([accept17, pd.get_dummies(accept17['gender'])], axis=1)
    return accept17

accept18 = pd.read_csv("test_dataset/accept18.csv")
rejected18 = pd.read_csv("test_dataset/rejected18.csv")
# rejected18 = get_frame("test_dataset/ICLR_2018_rejected.txt", 0)
# rejected18.to_csv("test_dataset/rejected18.csv")

test_dset = pd.concat([accept18, rejected18])
test_dset.to_csv("test_dataset/2018.csv")

accept17 = pd.read_csv("train_dataset/accept17.csv")
rejected17 = pd.read_csv("train_dataset/rejected17.csv")
# rejected18 = get_frame("test_dataset/ICLR_2018_rejected.txt", 0)
# rejected18.to_csv("test_dataset/rejected18.csv")

train_dset = pd.concat([accept17, rejected17])
train_dset.to_csv("train_dataset/2017.csv")
