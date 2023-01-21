import os
gender_d = {}
with open("../wsj0_gender_list.txt", 'r') as f_gender:
    lines = f_gender.readlines()
    for line in lines:
        speaker, gender = line.split()
        gender_d[speaker] = gender

def get_speaker(path):
    return os.path.split(os.path.dirname(path))[1]
        
with open("mix_2_spk_tr.txt", 'r') as f_train:
    lines = f_train.readlines()
    count = 0
    for line in lines:
        path1, _, path2, _ = line.split()
        g1 = gender_d[get_speaker(path1)]
        g2 = gender_d[get_speaker(path2)]
        if g1 == 'm' and g2 == 'm':
            print(get_speaker(path1), get_speaker(path2), path1, path2)
            count += 1
    print("MM:", count , "/" , len(lines))
        
    