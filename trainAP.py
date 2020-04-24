import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# FIXME: Normalization: Make the data more regular for calculation
def normalization(data, tag=""):
    mean = data.mean()
    maximum = data.max()
    minimum = data.min()
    print(tag, mean, maximum, minimum)
    return (data - mean) / (maximum - minimum)


# FIXME: Read the data
df = pandas.read_csv("APAO_pb.csv")

# FIXME: Clean the data, setting the value of the missing data to the median of the column
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values

for col in numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('inputting missing values for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing
        med = df[col].median()
        df[col] = df[col].fillna(med)

# Disrupt data
# df = shuffle(df)
# df = shuffle(df)

# FIXME: Separate data
Year = df['Year'].values
Year = normalization(Year)

G2 = df['2GAP'].values
G2 = normalization(G2)

G3 = df['3GAP'].values
G3 = normalization(G3)

G4 = df['4GAP'].values
NG4 = normalization(G4)

GT = df['TotalAP'].values
NGT = normalization(GT)

print(G2.shape, G3.shape, G4.shape, GT.shape)

# FIXME: 4G Prediction in Asia Pacific
data = np.array([Year, G2, NGT])
data = data.T
train_fraction = .83
train_number = int(df.shape[0] * train_fraction)
X_train = data[:train_number]
X_test = data[train_number:]
y_train = G4[:train_number]
y_test = G4[train_number:]

# model
clf = GridSearchCV(SVR(kernel='rbf', gamma=0.1), {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, cv=5)
clf.fit(X_train, y_train)
result = clf.score(X_train, y_train)
test = clf.score(X_test, y_test)

c = clf.best_params_
y = clf.predict(X_test)
x = df['Year'][train_number:]

print(clf.best_params_, result, test)

deviation = y - y_test
deviation = deviation.flatten()
deviation = abs(deviation)
print(np.median(y))

# FIXME: Total Prediction in Asia Pacific
data_T = np.array([Year, G2, G3, NG4])
data_T = data_T.T
train_fraction_T = .6
train_number_T = int(df.shape[0] * train_fraction_T)
X_train_T = data[:train_number_T]
X_test_T = data[train_number_T:]
y_train_T = GT[:train_number_T]
y_test_T = GT[train_number_T:]

# model
clf_T = GridSearchCV(SVR(kernel='rbf', gamma=0.1), {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, cv=5)
clf_T.fit(X_train_T, y_train_T)
result_T = clf_T.score(X_train_T, y_train_T)
test_T = clf_T.score(X_test_T, y_test_T)

c_T = clf_T.best_params_
y_T = clf_T.predict(X_test_T)
x_T = df['Year'][train_number_T:]

print(clf_T.best_params_, result_T, test_T)

deviation_T = y_T - y_test_T
deviation_T = deviation_T.flatten()
deviation_T = abs(deviation_T)
print(np.median(y_T[train_number-train_number_T:]))


# FIXME: 5G Prediction

df5 = pandas.read_csv("AP5G.csv")
df_numeric_5 = df5.select_dtypes(include=[np.number])
numeric_cols_5 = df_numeric_5.columns.values

Y5G = df5['Year5G'].values
Y5G = normalization(Y5G)
T5 = df5['Total'].values
T5 = normalization(T5)
G54 = df5['4G'].values
G54 = normalization(G54)
GT5 = df5['G5AP'].values

data5 = np.array([Y5G, GT5, T5, G54])
data5 = data5.T
train_fraction_5 = .6
train_number_5 = int(df5.shape[0] * train_fraction_5)
X_train_5 = data5[:train_number_5]
X_test_5 = data5[train_number_5:]
y_train_5 = GT5[:train_number_5]
y_test_5 = GT5[train_number_5:]

# model
clf5 = GridSearchCV(SVR(kernel='rbf', gamma=0.1), {"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, cv=3)
clf5.fit(X_train_5, y_train_5)
result_5 = clf5.score(X_train_5, y_train_5)
test_5 = clf5.score(X_test_5, y_test_5)

c_5 = clf5.best_params_
y_5 = clf5.predict(X_test_5)
x_5 = df5['Year5G'][train_number_5:]

print(clf5.best_params_, result_5, test_5)


'''
deviation_5 = deviation_T[train_number-train_number_T:] - deviation
deviation_5 = deviation_5.flatten()
deviation_5 = abs(deviation_5)
print(np.median(y_T[train_number-train_number_T:]-y))

# FIXME: Data visualization
fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="4G Prediction"),
              row=1, col=1)
fig.add_trace(go.Bar(x=[0, 1, 2, 3, 4], y=deviation, name="4G Deviation"),
              row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=y_test, mode="lines", name="4G"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=x_T, y=y_T, mode="lines", name="Total Prediction"),
              row=1, col=1)
fig.add_trace(go.Bar(x=[0, 1, 2, 3, 4], y=deviation_T, name="Total Deviation"),
              row=2, col=2)
fig.add_trace(go.Scatter(x=x_T, y=y_test_T, mode="lines", name="Total"),
              row=2, col=1)
fig.add_trace(go.Scatter(x=df['Season'][train_number:], y=y_T[train_number-train_number_T:]-y, mode="lines", name="5G Prediction"),
              row=3, col=1)
fig.add_trace(go.Bar(x=[0, 1, 2, 3, 4], y=deviation_5, name="5G Deviation"),
              row=3, col=2)
fig.add_trace(go.Scatter(x=df['Season'][train_number:], y=y_test_T[train_number-train_number_T:]-y_test, mode="lines", name="5G"),
              row=3, col=1)

fig.update_layout(title="Prediction: 4G 5G in Asia Pacific")

fig.show()

'''