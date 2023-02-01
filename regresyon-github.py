import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')

df_ = pd.read_csv("yl tez github/players_22.csv")
df = df_.copy()

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

############################################################################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

scaler = StandardScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

y = dff["overall"]
X = dff.drop(["overall", "potential"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=46)

############################################################################################################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

################################################################################################

print("mse")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    msg = "%s: (%f)" % (name, mse)
    print(msg)

print("rmse")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: (%f)" % (name, rmse)
    print(msg)

print("mae")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    msg = "%s: (%f)" % (name, mae)
    print(msg)

#######################################################################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

lgb_model = LGBMRegressor().fit(X_train, y_train)
plot_importance(lgb_model, X_train)

##########################################################################################

dff.drop(["work_rate_High/Low", "work_rate_High/Medium", "work_rate_Low/High", "skill_moves",
          "work_rate_Medium/Low", "body_type_Lean (185+)", "body_type_Normal (170-185)",
          "body_type_Normal (185+)", "body_type_Stocky (170-)", "body_type_Normal (170-)",
          "body_type_Stocky (185+)", "body_type_Unique", "real_face_Yes","body_type_Stocky (170-185)",
          "sofifa_id", "club_team_id", "nationality_id",
          "nation_team_id", "work_rate_Medium/High", "weak_foot", "work_rate_Medium/Medium",
          "work_rate_Low/Medium", "work_rate_Low/Low", "body_type_Lean (170-185)",
          "league_level", "value_eur", "wage_eur", "release_clause_eur"], axis=1, inplace=True)

##############################################################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

scaler = StandardScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

y = dff["overall"]
X = dff.drop(["overall", "potential"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=46)

#################################################################################################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

################################################################################################

print("mse")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    msg = "%s: (%f)" % (name, mse)
    print(msg)

print("rmse")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    msg = "%s: (%f)" % (name, rmse)
    print(msg)

print("mae")
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    msg = "%s: (%f)" % (name, mae)
    print(msg)
