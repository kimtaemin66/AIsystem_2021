from sklearn import datasets
from sklearn import svm

digit = datasets.load_digits()

s=svm.SVC(gamma=0.1, C=10)
s.fit(digit.data, digit.target)

new_d = [digit.data[0], digit.data[1], digit.data[2]]
results = s.predict(new_d)

print("예측값: ", results)
print("참값: ", digit.target[0], digit.target[1], digit.target[2])

results_2 = s.predict(digit.data)
correct = [i for i in range(len(results_2))if results_2[i] == digit.target[i]]
accuracy = len(correct)/len(results_2)
print("정확도: ", accuracy*100, "%")