import numpy as np
import pandas as pd
import torchvision.transforms as transforms

# Image transformation
transform_horizontal = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(1),
    # transforms.functional.crop(0,6,84,84),
    transforms.RandomHorizontalFlip(1),
    # transforms.CenterCrop(70),
    # transforms.Grayscale(1),
    # transforms.Normalize(0.5, 0.5),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.functional.crop(0,6,84,84),
    # transforms.CenterCrop(70),
    # transforms.Grayscale(1),
    # transforms.Normalize(0.5, 0.5),
    
])

df = pd.read_pickle("./original_data/dataset.pkl")
X = np.stack(df["state"], 0)
y = np.stack(df["action"], 0)
s = np.stack(df["speed"], 0)
frequency_dict = {}
for num in y:
    if num in frequency_dict:
        frequency_dict[num] += 1
    else:
        frequency_dict[num] = 1
print(frequency_dict)
# df = df.drop(df[df['action'] == 0].sample(1000).index)
df = df.drop(df[df['action'] == 3].sample(800).index)
# df = df.drop(df[df['action'] == 2].sample(1000).index)
# df_dagger = pd.read_pickle("./dagger_data/dagger_dataset.pkl")
# df_dagger = df_dagger.drop(df_dagger[df_dagger['action'] == 0].sample(300).index)
# df_dagger = df_dagger.drop(df_dagger[df_dagger['action'] == 2].sample(2000).index)
# df_dagger = df_dagger.drop(df_dagger[df_dagger['action'] == 3].sample(300).index)
# df = df.append(df_dagger, ignore_index=True)
X = np.stack(df["state"], 0)
y = np.stack(df["action"], 0)


aug_data = {
    "state": [],
    "action": [],
    "speed": [],
}
for obs, act, speed in zip(X,y, s):
    if act == 4: # brake
        for _ in range(4):
            aug_data["state"].append(transforms.functional.crop(transform(obs), 0,6,84,84))
            new_act = 4
            aug_data["action"].append(new_act)
            aug_data["speed"].append(speed)
    if act == 1: # Right
            aug_data["state"].append(transforms.functional.crop(transform_horizontal(obs), 0,6,84,84))
            new_act = 2
            aug_data["action"].append(new_act)
            aug_data["speed"].append(speed)
    elif act == 2: # Left
        aug_data["state"].append(transforms.functional.crop(transform_horizontal(obs), 0,6,84,84))
        new_act = 1
        aug_data["action"].append(new_act) 
        aug_data["speed"].append(speed)  
    aug_data["state"].append(transforms.functional.crop(transform(obs), 0,6,84,84))
    aug_data["action"].append(act)  
    aug_data["speed"].append(speed)

X,y,data = 0,0,0
frequency_dict = {}
for num in aug_data["action"]:
    if num in frequency_dict:
        frequency_dict[num] += 1
    else:
        frequency_dict[num] = 1

print(frequency_dict)

# Concatenate data


df = pd.DataFrame(aug_data)

df.to_pickle('./test_data/dataset.pkl')