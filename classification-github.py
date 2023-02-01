import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay,\
    confusion_matrix, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')

df_ = pd.read_csv("yl tez github/players_22.csv")

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

###########################################################################################################

df = df_.copy()

df.loc[(df["overall"]>=0) & (df["overall"]<=60),"Target"] = 0
df.loc[(df["overall"]>=61) & (df["overall"]<=70),"Target"] = 1
df.loc[(df["overall"]>=71) & (df["overall"]<=75),"Target"] = 2
df.loc[(df["overall"]>75),"Target"] = 3

y = df["Target"]

df.drop(["Target", "overall", "potential"], axis=1, inplace=True)

###########################################################################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
scaler = StandardScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

###########################################################################################################

dff_train, dff_test, y_train, y_test = train_test_split(dff, y, test_size=0.20, random_state=17)

###########################################################################################################
"""Bütün modeller önce dff-train, dff_test ile çalıştırılacak (base model),
sonra trainsubset, testsubset ile çalıştırılacak"""

workbook = pd.read_excel("yl tez github/nca_features.xlsx",header=None)
WS_np = np.array(workbook)
WS_np.shape
subset=WS_np.reshape(38,)
subset.shape
subset
subset
trainsubset=dff_train.iloc[:,subset]
testsubset=dff_test.iloc[:,subset]

#########################################################################################################

# Logistic regression

ltr = LogisticRegression()
_ = ltr.fit(dff_train, y_train)
y_pred = ltr.predict(dff_test)

print("logreg")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# LDA

lda= LinearDiscriminantAnalysis()
_ = lda.fit(dff_train, y_train)
y_pred = lda.predict(dff_test)

print("lda")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# Naive Bayes

nby= GaussianNB()
_ = nby.fit(dff_train, y_train)
y_pred = nby.predict(dff_test)

print("nby")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# KNeighborsClassifier

knn = KNeighborsClassifier()
_ = knn.fit(dff_train, y_train)
y_pred = knn.predict(dff_test)

print("KNN")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# SVC(gamma='auto')

svc = SVC()
_ = svc.fit(dff_train, y_train)
y_pred = svc.predict(dff_test)

print("SVC")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# DecisionTreeClassifier

dsc = DecisionTreeClassifier()
_ = dsc.fit(dff_train, y_train)
y_pred = dsc.predict(dff_test)

print("destree")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# RandomForestClassifier

rfc = RandomForestClassifier()
_ = rfc.fit(dff_train, y_train)
y_pred = rfc.predict(dff_test)

print("randforest")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# XGBClassifier

xgb = XGBClassifier()
_ = xgb.fit(dff_train, y_train)
y_pred = xgb.predict(dff_test)

print("xgb")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# LightGBM

lgbm = LGBMClassifier()
_ = lgbm.fit(dff_train, y_train)
y_pred = lgbm.predict(dff_test)

print("lgbm")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# CatBoostClassifier(verbose=False)

ctb = CatBoostClassifier()
_ = ctb.fit(dff_train, y_train)
y_pred = ctb.predict(dff_test)

print("catboost")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

#####################################################################################################

# Logistic regression

ltr = LogisticRegression()
_ = ltr.fit(trainsubset, y_train)
y_pred = ltr.predict(testsubset)

print("logreg")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# LDA

lda= LinearDiscriminantAnalysis()
_ = lda.fit(trainsubset, y_train)
y_pred = lda.predict(testsubset)

print("logreg")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# Naive Bayes

nby= GaussianNB()
_ = nby.fit(trainsubset, y_train)
y_pred = nby.predict(testsubset)

print("nby")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# KNeighborsClassifier

knn = KNeighborsClassifier()
_ = knn.fit(trainsubset, y_train)
y_pred = knn.predict(testsubset)

print("KNN")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# SVC(gamma='auto')

svc = SVC()
_ = svc.fit(trainsubset, y_train)
y_pred = svc.predict(testsubset)

print("SVC")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# DecisionTreeClassifier

dsc = DecisionTreeClassifier()
_ = dsc.fit(trainsubset, y_train)
y_pred = dsc.predict(testsubset)

print("destree")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# RandomForestClassifier

rfc = RandomForestClassifier()
_ = rfc.fit(trainsubset, y_train)
y_pred = rfc.predict(testsubset)

print("randforest")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# XGBClassifier

xgb = XGBClassifier()
_ = xgb.fit(trainsubset, y_train)
y_pred = xgb.predict(testsubset)

print("xgb")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# LightGBM

lgbm = LGBMClassifier()
_ = lgbm.fit(trainsubset, y_train)
y_pred = lgbm.predict(testsubset)

print("lgbm")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))

# CatBoostClassifier(verbose=False)

ctb = CatBoostClassifier()
_ = ctb.fit(trainsubset, y_train)
y_pred = ctb.predict(testsubset)

print("catboost")
print(classification_report(y_test, y_pred, digits=4))
print(cohen_kappa_score(y_test, y_pred))







