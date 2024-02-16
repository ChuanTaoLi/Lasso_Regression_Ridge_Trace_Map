import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

'''加载数据集'''
data = pd.read_excel(r'D:\01代码收集\Lasso\NSLKDD_All.xlsx')
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
train = pd.read_csv(r'D:\01代码收集\Lasso\train_data.csv')
test = pd.read_csv(r'D:\01代码收集\Lasso\test_data.csv')

'''定义lasso回归函数'''
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score


def lasso_regression(train, test, alpha):
    lassoreg = Lasso(alpha=alpha, max_iter=1000, fit_intercept=True)
    lassoreg.fit(train.iloc[:, :-1], train['label'])
    feature_count = np.sum(lassoreg.coef_ != 0)  # 计算该alpha下筛选出的特征数量
    y_pred = lassoreg.predict(test.iloc[:, :-1])

    y_pred_classes = np.round(y_pred).astype(int)
    ret = [alpha, accuracy_score(test['label'], y_pred_classes)]
    ret.append(feature_count)  # 非零系数的数量
    ret.extend(lassoreg.coef_)

    return ret


'''记录不同alpha下的准确率以及各特征的回归系数'''
alpha_lasso = np.linspace(0, 0.6, 1000)
# 定义coef_matrix_lasso每列的标签，分别为alpha、准确率以及数据集的各特征名称
col = ["alpha", "accuracy", "feature_count"] + list(X.columns)
# 将采样的alpha数值插入到占位符上
ind = ["alpha_%.4g" % alpha_lasso[i] for i in range(0, len(alpha_lasso))]
# 存储Lasso回归不同alpha值下的准确率和系数的DataFrame
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
# 调用函数，将函数返回的结果录入
for i in range(len(alpha_lasso)):
    coef_matrix_lasso.iloc[i,] = lasso_regression(train, test, alpha_lasso[i])
coef_matrix_lasso.to_excel(r'coef_matrix_lasso.xlsx', index=True)

'''绘制岭迹图'''
plt.figure(figsize=(14, 6.8))
for i in np.arange(len(list(X.columns))):
    plt.plot(coef_matrix_lasso["alpha"], coef_matrix_lasso[list(X.columns)[i]],
             color=plt.cm.Set1(i / len(list(X.columns))), label=list(X.columns)[i])
    plt.legend(loc="upper right", ncol=4)
    plt.xlabel("Alpha", fontsize=14)
    plt.ylabel("Coefficient", fontsize=14)
    plt.title('Lasso Regression Ridge Trace Map', fontsize=16)
plt.savefig(r'Lasso Regression Ridge Trace Map.png', dpi=600)
plt.show()
