import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

df_NA = pd.read_csv('NAAO_pb.csv')
df_numeric_NA = df_NA.select_dtypes(include=[np.number])
numeric_cols_NA = df_numeric_NA.columns.values

df_EMEA = pd.read_csv('EMEAAO_pb.csv')
df_numeric_EMEA = df_EMEA.select_dtypes(include=[np.number])
numeric_cols_EMEA = df_numeric_EMEA.columns.values

df_AP = pd.read_csv('APAO_pb.csv')
df_numeric_AP = df_AP.select_dtypes(include=[np.number])
numeric_cols_AP = df_numeric_AP.columns.values

df_CALA = pd.read_csv('CALAAO_pb.csv')
df_numeric_CALA = df_CALA.select_dtypes(include=[np.number])
numeric_cols_CALA = df_numeric_CALA.columns.values

fig = make_subplots(rows=4, cols=4)

fig.add_trace(go.Box(y=df_NA['TotalNA'], name="Total in North America"), row=1, col=1)
fig.add_trace(go.Box(y=df_AP['TotalAP'], name="Total in Asia Pacific"), row=1, col=2)
fig.add_trace(go.Box(y=df_EMEA['TotalEMEA'], name="Total in Europe, Middle East and Africa"), row=1, col=3)
fig.add_trace(go.Box(y=df_CALA['TotalCALA'], name="Total in Caribbean And Latin America"), row=1, col=4)

fig.add_trace(go.Box(y=df_NA['4GNA'], name="4G in North America"), row=2, col=1)
fig.add_trace(go.Box(y=df_AP['4GAP'], name="4G in Asia Pacific"), row=2, col=2)
fig.add_trace(go.Box(y=df_EMEA['4GEMEA'], name="4G in Europe, Middle East and Africa"), row=2, col=3)
fig.add_trace(go.Box(y=df_CALA['4GCALA'], name="4G in Caribbean And Latin America"), row=2, col=4)

fig.add_trace(go.Box(y=df_NA['3GNA'], name="3G in North America"), row=3, col=1)
fig.add_trace(go.Box(y=df_AP['3GAP'], name="3G in Asia Pacific"), row=3, col=2)
fig.add_trace(go.Box(y=df_EMEA['3GEMEA'], name="3G in Europe, Middle East and Africa"), row=3, col=3)
fig.add_trace(go.Box(y=df_CALA['3GCALA'], name="3G in Caribbean And Latin America"), row=3, col=4)

fig.add_trace(go.Box(y=df_NA['2GNA'], name="2G in North America"), row=4, col=1)
fig.add_trace(go.Box(y=df_AP['2GAP'], name="2G in Asia Pacific"), row=4, col=2)
fig.add_trace(go.Box(y=df_EMEA['2GEMEA'], name="2G in Europe, Middle East and Africa"), row=4, col=3)
fig.add_trace(go.Box(y=df_CALA['2GCALA'], name="2G in Caribbean And Latin America"), row=4, col=4)

fig.update_layout(title="Exception of 2G, 3G, 4G and Total")
fig.show()