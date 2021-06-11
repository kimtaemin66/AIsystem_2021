from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

digit = datasets.load_digits()
s = svm.SVC(gamma = 0.001)
accuracies = cross_val_score(s, digit.data, digit.target, cv = 20)

print(accuracies)
print("평균정확률: ", accuracies.mean()*100, "%")