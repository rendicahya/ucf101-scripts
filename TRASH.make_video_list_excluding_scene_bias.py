train_list = open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101/ucf101_train_split_1_videos.txt",
    "r",
).readlines()

val_list = open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101/ucf101_val_split_1_videos.txt",
    "r",
).readlines()

scene_bias_list = open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101-scenes/ucf101_scenes_all.txt",
    "r",
).readlines()

scene_bias_list = [i.replace(".mp4", ".avi") for i in scene_bias_list]
debiased_train_list = [i for i in train_list if i not in scene_bias_list]
debiased_val_list = [i for i in val_list if i not in scene_bias_list]

with open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101/ucf101_train_split_1_videos_debiased.txt",
    "w",
) as w:
    w.writelines(debiased_train_list)

with open(
    "/nas.dbms/randy/projects/mmaction2/data/ucf101/ucf101_val_split_1_videos_debiased.txt",
    "w",
) as w:
    w.writelines(debiased_val_list)

print("Train size:", len(train_list))
print("Val size:", len(val_list))
print("Scene bias size:", len(scene_bias_list))
print("Debiased train size:", len(debiased_train_list))
print("Debiased val size:", len(debiased_val_list))
