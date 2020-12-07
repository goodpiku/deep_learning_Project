import json

if __name__ == '__main__':
    new_data={}
    with open('../nlu_traindev/train.json')as json_file:
        data = json.load(json_file)
    for key in data:
        if int(key)<=1000:
            if key not in new_data:
                new_data[key]=0
            new_data[key]=data[key]
    with open ('../nlu_traindev/less_train.json','w')as out_file:
        json.dump(new_data,out_file)



