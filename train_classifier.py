import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score #,confusion_matrix
import seaborn as sns
#import matplotlib.pyplot as plt
from tabulate import tabulate

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Preprocess data to ensure consistent shape
processed_data = []
max_sequence_length = 0

for data_point in data_dict['data']:
    if isinstance(data_point, list) and all(isinstance(x, (int, float)) for x in data_point):

        sequence_length = len(data_point)
        if sequence_length > max_sequence_length:
            max_sequence_length = sequence_length
        processed_data.append(data_point)
    else:
        print(f"Invalid data_point encountered: {data_point}")

# Pad sequences to the maximum sequence length
padded_data = []
for data_point in processed_data:
    padded_data_point = data_point + [0] * (max_sequence_length - len(data_point))
    padded_data.append(padded_data_point)

# Convert data and labels to NumPy arrays
data = np.array(padded_data)
labels = np.array(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

# Predict on test set
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print('{:.2f}% of samples were classified correctly!'.format(score * 100))

# # Generate confusion matrix
# conf_matrix = confusion_matrix(y_test, y_predict)

# # Plot confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

# Save the trained model using pickle
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
