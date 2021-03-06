import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objs as go
import pandas as pd




def plot_features_importance(data, importances, n_feat):
    """Plot the features importance barplot.

    Parameters
    ----------
    data : pd.DataFrame
        data containing colnames used in the model.

    importances : np.ndarray
        list of feature importances

    n_feat : int
        number of features to plot

    """
    indices = np.argsort(importances)[::-1]
    features = data.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices][:n_feat],
                y=features[indices][:n_feat], palette='Blues_r')
    plt.title("Top {} Features Importance".format(n_feat))
    return plt.show()


def plot_observation_contribution(interpretable_dict,
                                  index_ref,
                                  num_features_plot=5,
                                  is_horizontal=False
                                  ):
    """Compute plotly figure of contributions for one explanation
    (local interpretability).

    Parameters
    ----------
    interpretable_dict : dict,
        Dictionnary with post-processed data and contributions
        (see .postprocessing module for more information).

    index_ref : int,
        Number of line of DataFrame that should be plot.

    num_features : int, optional
        Number of features that should be plot (ordered by decreasing
        contribution).

    is_horizontal : bool, optional
        True if figure should be plot horizontally, False otherwise.

    Returns
    -------
    fig : plotly figure,
        Plotly figure of contributions.
    """

    contrib = interpretable_dict['contrib'].iloc[index_ref,:]
    bias = interpretable_dict['bias']
    pred = interpretable_dict['pred'][index_ref]

    contribution = pd.DataFrame(
        interpretable_dict['contrib'].columns.tolist(),
        columns=['variable'])
    contribution['contribution'] = contrib.values

    contribution = contribution.assign(sortval = np.abs(
        contribution.contribution))\
        .sort_values('sortval', ascending=False)\
        .drop('sortval', 1)

    contribution = contribution.iloc[:num_features_plot,:].append({
        contribution.columns[0]: '_OTHERS_',
        contribution.columns[1]: sum(contribution.iloc[num_features_plot:,1])
    }, ignore_index=True)

    data, layout = compute_waterfall_contribution(
        contribution,
        bias,
        delta_y_display=0.02)

    if is_horizontal:
        data, layout = make_waterfall_horizontal(data, layout)

    fig = go.Figure(data=data, layout=layout)
    return fig


def compute_waterfall_contribution(contribution, bias,
                                   n_round=3, delta_y_display=0.05,
                                   hover_value='x'):
    """Function used to compute plotly traces for a waterfall display.
    """

    ## Compute the 4 traces list
    base = [0]
    positive = [bias]
    negative = [0]
    total = [0]
    text = [str(round(bias, n_round))]
    y_text = [bias + delta_y_display]

    for contrib in contribution['contribution']:
        base.append(base[-1] + positive[-1])
        total.append(0)
        if contrib>=0:
            negative.append(0)
            positive.append(contrib)
            text.append('+' + str(round(contrib, n_round)))
        else:
            positive.append(0)
            negative.append(-contrib)
            base[-1] = base[-1] + contrib
            text.append(str(round(contrib, n_round)))

        y_text.append(base[-1] + negative[-1] + positive[-1] + delta_y_display)

    total.append(base[-1] + positive[-1])
    base.append(0)
    positive.append(0)
    negative.append(0)
    text.append(str(round(total[-1] + negative[-1], n_round)))
    y_text.append(total[-1] + delta_y_display)

    ## Create the 4 traces
    x_data = ['_BASE RATE_'] + list(contribution['variable']) + ['Prediction']
    trace_base = go.Bar(x=x_data, y=base,
    marker=dict(
        color='rgba(1,1,1, 0.0)',
    ),
    hoverinfo=hover_value)
    trace_positive = go.Bar(x=x_data, y=positive,
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    ),
    hoverinfo=hover_value)
    trace_negative = go.Bar(x=x_data, y=negative,
    marker=dict(
        color='rgba(255, 128, 0, 0.7)',
        line=dict(
            color='rgba(255, 128, 0, 1.0)',
            width=2,
        )
    ),
    hoverinfo=hover_value)
    trace_total = go.Bar(x=x_data, y=total,
    marker=dict(
        color='rgba(50, 171, 96, 0.7)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=2,
        )
    ),
    hoverinfo=hover_value)

    data = [trace_base, trace_positive, trace_negative, trace_total]


    annotations = []
    for i in range(len(text)):
        annotations.append(dict(x=x_data[i], y=y_text[i], text=text[i],
                                  font=dict(family='Arial', size=14,
                                  color='rgba(0, 0, 0, 1)'),
                                  showarrow=False,))

    layout = go.Layout(
                barmode='stack',
                xaxis={'title': ''},
                yaxis={'title': 'Prediction score'#,
                    #'range': [0.0, 1.1]
                    },
                title='Score breakdown by variable contribution',
                margin=go.layout.Margin(
                    l=200,
                    r=20,
                    b=100,
                    t=50,
                    pad=4
                ),
                #paper_bgcolor='rgba(245, 246, 249, 1)',
                #plot_bgcolor='rgba(245, 246, 249, 1)',
                showlegend=False
            )
    layout['annotations'] = annotations

    return data, layout


def make_waterfall_horizontal(data, layout):
    """Function used to flip the figure from vertical to horizontal.
    """
    h_data = list(data)
    h_data = []
    for i_trace, trace in enumerate(list(data)):
        h_data.append(trace)
        prov_x = h_data[i_trace]['x']
        h_data[i_trace]['x'] = list(h_data[i_trace]['y'])[::-1]
        h_data[i_trace]['y'] = list(prov_x)[::-1]
        h_data[i_trace]['orientation'] = 'h'
        h_data[i_trace]['hoverinfo'] = hoverinfo_horizontal_(
            h_data[i_trace]['hoverinfo'])

    h_annotations = []
    for i_ann, annotation in enumerate(list(layout['annotations'])):
        h_annotations.append(annotation)
        prov_x = h_annotations[i_ann]['x']
        h_annotations[i_ann]['x'] = h_annotations[i_ann]['y']
        h_annotations[i_ann]['y'] = prov_x
    h_annotations.reverse()

    h_layout = layout
    h_layout['annotations'] = h_annotations
    h_layout['xaxis'] = go.layout.XAxis({'title': 'Prediction score'})
    h_layout['yaxis'] = go.layout.YAxis({'title': ''})

    return h_data, h_layout


def hoverinfo_horizontal_(hover_value):
    """Function used to update the hoverinfo relevantly when the
    plot is made horizontal.
    """
    if hover_value == "x":
        horizontal_hover_value = "y"
    if hover_value == "y":
        horizontal_hover_value = "x"
    if hover_value == "x+text":
        horizontal_hover_value = "y+text"
    if hover_value == "y+text":
        horizontal_hover_value = "x+text"
    if hover_value == "text":
        horizontal_hover_value = "text"
    return horizontal_hover_value