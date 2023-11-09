import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os

#Example shape
# Wall time,Step,Value
# 1698314503.5367975,1,0.6933741569519043

val_loss = pd.read_csv('val_loss.csv')
train_loss = pd.read_csv('train_loss.csv')

val_loss = val_loss.drop(['Wall time'], axis=1)
train_loss = train_loss.drop(['Wall time'], axis=1)

# plot validation and training loss in the same plot

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=val_loss['Step'], y=val_loss['Value'],
                         mode='lines+markers',
                         name='Validation loss',
                         line=dict(color='blue', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=train_loss['Step'], y=train_loss['Value'],
                         mode='lines+markers',
                         name='Training loss',
                         line=dict(color='red', width=2)))

# Update layout
fig.update_layout(
    title={
        'text': 'Training and Validation Loss',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    },
    xaxis=dict(
        dtick=5,
        title='Epoch',
        titlefont_size=16,
        tickfont_size=14,
        gridcolor='gray',
        gridwidth=1,
    ),
    yaxis=dict(
        title='Loss',
        titlefont_size=16,
        tickfont_size=14,
        gridcolor='gray',
        gridwidth=1,
    ),
    legend=dict(
        x=0,
        y=1,
        traceorder='normal',
        font=dict(
            size=12,
        ),
    )
)

# Show the figure

fig.write_image("loss.pdf")
fig.show()







