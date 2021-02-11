import json
import string

punctuations = set(string.punctuation)


def my_data():
    with open('../benchmark/test.json', 'r') as j_file:
        dev_set = json.load(j_file)
    return dev_set


def read_answers_from_file(f):
    in_file = open(f, "r")
    out_file = open(f.replace('.in', '.out'), "r")
    intent_file = open(f.replace('seq.in', 'label'), 'r')
    list_of_lines = []
    for line in in_file:
        l = line.lstrip().rstrip()
        slots = out_file.readline().lstrip().rstrip()
        intent = intent_file.readline().lstrip().rstrip()
        list_of_lines.append((l, slots, intent))
    return list_of_lines


tr_lines = read_answers_from_file("../benchmark/answers/train/seq.in")
ts_lines = read_answers_from_file("../benchmark/answers/test/seq.in")
dv_lines = read_answers_from_file("../benchmark/answers/dev/seq.in")

all_answer_lines = tr_lines + ts_lines + dv_lines

# print(dev_set)
dev_set = my_data()

dict_of_findings = {}
count = 0
for key, value_dict in dev_set.items():
    text = value_dict['text'].lstrip().rstrip()
    # print(dev_set[])
    for ind, val in enumerate(all_answer_lines):
        txt, slots, intent = val
        if txt.lower() in text.lower():
            value_dict['intent'] = intent
            dict_of_findings[ind] = txt
            # if txt.lower() == text.lower():
            dict_of_findings[ind] = txt
            list_txt = txt.split(' ')
            slot = slots.split(' ')
            dict_slot = {}
            for i in range(len(slot)):
                if slot[i].find('B-') != -1:
                    dict_slot[slot[i][2:]] = list_txt[i]
                if slot[i].find('I-') != -1:
                    dict_slot[slot[i][2:]] = f'{dict_slot[slot[i][2:]]} {list_txt[i]}'
            value_dict['slots'] = dict_slot
print(len(dict_of_findings))

# print(dict_of_findings)
# print(len(dict_of_findings))
with open('../benchmark/j_new_test.json', 'w') as l:
    json.dump(dev_set, l)
