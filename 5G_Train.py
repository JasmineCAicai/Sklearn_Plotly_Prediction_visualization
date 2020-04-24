import trainCALA as tc
import trainEMEA as te

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=tc.df['Season'][-4:], y=[11.56, 12.30, 13.00, 13.73],
                         mode="lines", name="北美"), row=1, col=1)
fig.add_trace(go.Scatter(x=tc.df['Season'][-4:], y=[22.88, 24.28, 25.60, 26.99],
                         mode="lines", name="亚太"), row=1, col=1)
fig.add_trace(go.Scatter(x=tc.df['Season'][-4:], y=tc.y_T[-4:]-tc.y[-4:],
                         mode="lines", name="CALA"), row=1, col=1)
fig.add_trace(go.Scatter(x=te.df['Season'][-4:], y=te.y_T[-4:]-te.y[-4:],
                         mode="lines", name="EMEA"), row=1, col=1)

fig.update_layout(title="5G规模预测")

fig.show()
