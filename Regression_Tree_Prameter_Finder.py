import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from matplotlib import font_manager


def best_find1():
    df_raw = pd.read_csv('Final.csv')
    X1 = df_raw.loc[:, ['Make']]
    X1=pd.get_dummies(X1,columns=["Make"])   #get_dummies对“整数特征”无变化，对“类别特征”one-hot编码
    X1=X1.values
    X2=df_raw.loc[:, ['Length (ft)','YearDis','IsCatamarans','LOA','LWL','Beam','SA','Draft','Displacement','GDP','Gini']].values
    X = np.hstack([X2,X1])
    y= df_raw.loc[:, 'Listing Price (USD)'].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=60, train_size=0.9)
    train_score=[]
    test_score=[]
    #在此设置参数查询范围
    begin,end=30,50
    for deep in range(begin,end):
        forest1 = RandomForestRegressor(n_estimators=150,max_depth=deep)#n_estimators：随机森林中树的棵数；
                                                                        #max_depth：每棵树的最大深度。
        forest1.fit(x_train, y_train)
        tc1=forest1.score(x_train, y_train)
        tc2=forest1.score(x_test, y_test)
        train_score.append(tc1)
        test_score.append(tc2)
    plt.figure()
    plt.plot(np.arange(begin,end), train_score, "go-", label="Train")
    plt.plot(np.arange(begin,end), test_score, "ro-", label="Test")
    plt.legend()
    plt.show()
best_find1()