import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def train_knn():

    print("knn file start!")

    data_g=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_jirak.xls')
    data_l=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature_seongan.xls')
    data_n=pd.read_excel('C:/Users/user/Desktop/psl/feature_normal.xls')
    data_p=pd.read_excel('C:/Users/user/Desktop/psl/fault_feature 3L_100.xls')

    df = pd.concat([data_g, data_l, data_n,data_p])

    target = []

    for idx, row in df.iterrows():
        if row['type'] == 0:
            target.append(0)
        elif row['type'] == 1 or row['type'] == 2 or row['type'] == 3:
            target.append(1)
        elif row['type'] == 4 or row['type'] == 5 or row['type'] == 6:
            target.append(2)
        else:
            target.append(3)

    df['type'] = target

    y = df.filter(['type']) 
    X = df.drop(['target','type','m'], axis=1)

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X, y)

    joblib.dump(knn,'./knn.pkl')

    print("knn file complete!")
    
if __name__ == "__main__":
    train_knn()