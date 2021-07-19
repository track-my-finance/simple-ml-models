from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics

IRIS_DATASET = "iris.csv"

iris = pd.read_csv(IRIS_DATASET)

train, test = train_test_split(iris, test_size=0.3)

train_x = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
train_y = train.species

test_x = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
test_y = test.species

model = LinearDiscriminantAnalysis()
classifier = model.fit(train_x, train_y)
prediction = model.predict(test_x)

# Metrics calculations
accuracy = metrics.accuracy_score(prediction, test_y) * 100
precision_score = metrics.precision_score(
    prediction, test_y, average=None) * 100
confusion_matrix = metrics.confusion_matrix(prediction, test_y)

confusion_matrix_diagonal = np.diag(confusion_matrix)

FP = confusion_matrix.sum(axis=0) - confusion_matrix_diagonal
FN = confusion_matrix.sum(axis=1) - confusion_matrix_diagonal
TP = confusion_matrix_diagonal
TN = confusion_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
f1_score = 2*TP/(2*TP + FP + FN)

class_labels = ["setosa", "versicolor", "virginica"]

print(f'Accuracy: {accuracy}')

rows = ["Precision", "True positives", "True Negatives", "False Positives",
        "False Negatives", "Sensitivity", "Specificity", "F1 Score"]


metrics_table: dict = {}

for i in range(len(class_labels)):
    attributes: list = []
    attributes.append(precision_score[i])
    attributes.append(TP[i])
    attributes.append(TN[i])
    attributes.append(FP[i])
    attributes.append(FN[i])
    attributes.append(sensitivity[i]*100)
    attributes.append(specificity[i]*100)
    attributes.append(f1_score[i]*100)

    series = pd.Series(attributes, index=rows)

    metrics_table[class_labels[i]] = series

metrics_dataframe = pd.DataFrame(metrics_table)

metrics_dataframe.loc["Precision",
                      :] = metrics_dataframe.loc["Precision", :].astype(str) + "%"
metrics_dataframe.loc["Sensitivity",
                      :] = metrics_dataframe.loc["Sensitivity", :].astype(str) + "%"
metrics_dataframe.loc["Specificity",
                      :] = metrics_dataframe.loc["Specificity", :].astype(str) + "%"
metrics_dataframe.loc["F1 Score",
                      :] = metrics_dataframe.loc["F1 Score", :].astype(str) + "%"


print(metrics_dataframe)

confusion_matrix_display = metrics.plot_confusion_matrix(
    classifier, test_x, test_y, display_labels=class_labels, cmap=plt.cm.Blues)
plt.show()
