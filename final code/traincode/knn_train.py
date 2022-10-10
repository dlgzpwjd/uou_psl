import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_knn(path):

    print("knn file start!")

    data_g=pd.read_excel(path + '/dataset/train 1L 1000.xlsx')
    data_l=pd.read_excel(path + '/dataset/train 2L 1000.xls')
    data_n=pd.read_excel(path + '/dataset/train nomal 1000.xlsx')
    data_p=pd.read_excel(path + '/dataset/train 3L 300.xls')

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

    joblib.dump(knn, path + '/model/knn.pkl')

    print("knn file complete!")
    
if __name__ == "__main__":
    train_knn()