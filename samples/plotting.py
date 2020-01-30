import numpy as np
import os

import plotly
import plotly.graph_objects as go

def plot_observer_error(true, estimate):
    dims = np.shape(true)
    xs = np.arange(dims[0])
    fig = plotly.subplots.make_subplots(rows=1, cols=1,
                                        subplot_titles=("Observer Error per Dim"),
                                        vertical_spacing=.15)  # go.Figure()

    for i in range(dims[1]):
        fig.add_trace(go.Scatter(x=xs, y=true[:,i], name=f"Dim {i} true",
                                 line=dict(color='red',width=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=estimate[:, i], name=f"Dim {i} est",
                                 line=dict(color='blue', width=4)), row=1, col=1)


    fig.update_layout(title='Observer Convergence',
                      xaxis_title='Timestep',
                      yaxis_title='Unit',
                      plot_bgcolor='white',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      )
    fig.show()

def plot_rollout(states, actions):
    ar = np.stack(states)
    l = np.shape(ar)[0]
    xs = np.arange(l)

    yaw = ar[:, 0]
    pitch = ar[:, 1]
    roll = ar[:, 2]

    actions = np.stack(actions)

    fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                        subplot_titles=("Euler Angles", "Actions"),
                                        vertical_spacing=.15)  # go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=yaw, name='Yaw',
                             line=dict(color='firebrick', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=pitch, name='Pitch',
                             line=dict(color='royalblue', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=roll, name='Roll',
                             line=dict(color='green', width=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=xs, y=actions[:, 0], name='M1',
                             line=dict(color='firebrick', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 1], name='M2',
                             line=dict(color='royalblue', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 2], name='M3',
                             line=dict(color='green', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 3], name='M4',
                             line=dict(color='orange', width=4)), row=2, col=1)

    fig.update_layout(title='Euler Angles from MPC Rollout',
                      xaxis_title='Timestep',
                      yaxis_title='Angle (Degrees)',
                      plot_bgcolor='white',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      )
    fig.show()