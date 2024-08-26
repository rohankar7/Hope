import os
import pandas as pd

file_path = 'C:/Project/GPTImages'

ShapeNetCoreDescriptions = {
    'Class': [],
    'Subclass': [],
    'Description': []
}

for classes in os.listdir(file_path):
    class_path = os.path.join(file_path, classes)
    path = os.path.join(class_path, 'desc.txt')
    if os.path.isfile(path):
        descriptions = None
        with open(path, 'r') as f:
            descriptions = [line for line in f]
        f.close()
        for i in range(len(descriptions)):
            if i%2==0:
                ShapeNetCoreDescriptions['Class'].append(str(classes))
                ShapeNetCoreDescriptions['Subclass'].append(descriptions[i].split(':')[0])
            else:
                ShapeNetCoreDescriptions['Description'].append(str(descriptions[i][0:-1][76:].split('\n')[0]))
print(len(ShapeNetCoreDescriptions['Class']))
print(len(ShapeNetCoreDescriptions['Subclass']))
print(len(ShapeNetCoreDescriptions['Description']))

df = pd.DataFrame(ShapeNetCoreDescriptions)
df.to_csv("./descriptions.csv", index=False)