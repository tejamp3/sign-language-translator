import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

print("Inspecting data shapes...")
for i, item in enumerate(data_dict['data']):
    print(f"Item {i} shape: {np.shape(item)}")

max_length = max(len(item) for item in data_dict['data'])
padded_data = [
    np.pad(item, (0, max_length - len(item)), 'constant') if len(item) < max_length else item
    for item in data_dict['data']
]
data = np.asarray(padded_data)

print(f"Data shape after padding: {data.shape}")

labels = np.asarray(data_dict['labels'])

if len(data) != len(labels):
    raise ValueError(f"Mismatch: {len(data)} samples and {len(labels)} labels.")

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

