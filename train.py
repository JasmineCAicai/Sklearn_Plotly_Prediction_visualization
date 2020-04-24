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
df = pandas.read_csv("NAAO_pb_c.csv")

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

G2 = df['2GNA'].values
G2 = normalization(G2)

G3 = df['3GNA'].values
G3 = normalization(G3)

G4 = df['4GNA'].values
NG4 = normalization(G4)

GT = df['TotalNA'].values
NGT = normalization(GT)

print(G2.shape, G3.shape, G4.shape, GT.shape)

# FIXME: 4G Prediction in North America
data = np.array([Year, G2, NGT])
data = data.T
train_fraction = .57
train_number = int(df.shape[0] * train_fraction)
X_train = data[:train_number]
X_test = data[train_number:]
y_train = G4[:train_number]
y_test = G4[train_number:]

print(np.max(G4))

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
print(np.median(deviation))

# FIXME: Total Prediction in North America
data_T = np.array([Year, G2, G3, NG4])
data_T = data_T.T
train_fraction_T = .8
train_number_T = int(df.shape[0] * train_fraction_T)
X_train_T = data[:train_number_T]
X_test_T = data[train_number_T:]
y_train_T = GT[:train_number_T]
y_test_T = GT[train_number_T:]

print(np.max(GT))

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
print(np.median(deviation_T))

deviation_5 = deviation[train_number_T-train_number:] - deviation_T
deviation_5 = deviation_5.flatten()
deviation_5 = abs(deviation_5)

'''
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
fig.add_trace(go.Scatter(x=df['Season'][train_number_T:], y=y[train_number_T-train_number:]-y_T,
                         mode="lines", name="5G Prediction"), row=3, col=1)
fig.add_trace(go.Bar(x=[0, 1, 2, 3, 4], y=deviation_5, name="5G Deviation"),
              row=3, col=2)
fig.add_trace(go.Scatter(x=df['Season'][train_number_T:], y=y_test[train_number_T-train_number:]-y_test_T,
                         mode="lines", name="5G"), row=3, col=1)

fig.update_layout(title="Prediction: 4G 5G in North America")

fig.show()
'''
