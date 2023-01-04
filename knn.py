from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
irisData=load_iris()
a=irisData.data
b=irisData.target
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.7,random_state=6)
Knn=KNeighborsClassifier(n_neighbors=7)
Knn.fit(a_train,b_train)
c=Knn.predict(a_test)
acc=accuracy_score(b_test,c)
print(c)
print(acc)