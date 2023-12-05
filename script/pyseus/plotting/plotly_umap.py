import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.offline
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import umap


def scale_table(matrix, method):
    """
    takes a feature table and scale the data accordingly
    """


    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'min_max':
        scaler = MinMaxScaler()
    elif method == 'l1_norm':
        scaler = Normalizer(norm='l1')
    elif method == 'l2_norm':
        scaler = Normalizer(norm='l2')
    else:
        # if there is no specified scaler, return values with nulls dropped
        return matrix

    scaled = scaler.fit_transform(matrix)

    return scaled



def interaction_umap(
        matrix, node_name, cluster, x='umap_1', y='umap_2', opacity=0.7,
        width=800, height=600, highlight=None, unlabelled_color='#D0D3D4',
        unlabelled_opacity=0.1, hover_data=None, unlabelled_hover=True, search=False,
        categorical=True, na_in_data=True, highlight_color="red", pointsize=6):

    matrix = matrix.copy()
    matrix.reset_index(inplace=True, drop=False)


    if node_name == 'None':
        node_name = None
    if 'umap' in x:
        label_x = 'UMAP 1'
        label_y = 'UMAP 2'
    else:
        label_x = x
        label_y = y

    if cluster == 'None':
        fig = px.scatter(
            matrix,
            x=x,
            y=y,
            labels={
                x: label_x,
                y: label_y
            },
            hover_name=node_name,
            hover_data=hover_data,
            opacity=opacity,
            custom_data=['index'],
            template='simple_white')
        fig.update_traces(marker=dict(size=5.5))

    else:
        if na_in_data:
            labelling = matrix[cluster].isna()
            labelled = matrix[~labelling]
            if categorical:
                labelled[cluster] = labelled[cluster].astype(str)
            unlabelled = matrix[labelling]
            unlabelled[cluster] = 'unlabelled'
        else:
            labelled = matrix
            if categorical:
                labelled[cluster] = labelled[cluster].astype(str)
            unlabelled = matrix


        labelled.sort_values(by=cluster, inplace=True)

        fig2 = px.scatter(
            unlabelled,
            x=x,
            y=y,
            labels={
                x: label_x,
                y: label_y
            },
            color=cluster,
            hover_name=node_name,
            opacity=unlabelled_opacity,
            hover_data=hover_data,
            custom_data=['index'],
            color_discrete_sequence=[unlabelled_color],
            template='simple_white',
            render_mode = 'webgl')
        if not unlabelled_hover:
            fig2.update_traces(hoverinfo='skip', hovertemplate=None)
        fig2.update_traces(marker=dict(size=pointsize))

        #Lp = px.colors.qualitative.Light24
        #Dp = px.colors.qualitative.Dark24
        #Ap = px.colors.qualitative.Alphabet

        #ColPalDuo = px.colors.qualitative.D3 + [Lp[1] , Lp[4] , Dp[5] , Lp[17] , Dp[14] , Dp[7] , Lp[22], Ap[24] , Ap[10], Ap[0], Lp[0], Dp[9], Dp[6] ]
        # Duo: expand the color palette # for more options, see https://plotly.com/python/discrete-color/
        ColPalDuo = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10 + px.colors.qualitative.T10
        to_remove = ["#D62728", "#EF553B"]
        ColPalDuo = [x for x in ColPalDuo if x not in to_remove]

        fig1 = px.scatter(
            labelled,
            x=x,
            y=y,
            labels={
                x: label_x,
                y: label_y
            },
            hover_name=node_name,
            color=cluster,
            #color_continuous_scale=px.colors.cyclical.mygbm[: -1], # small palette (Kibeom)
            #color_discrete_sequence=px.colors.qualitative.Dark24, # Duo: expand the color palette # for more options, see https://plotly.com/python/discrete-color/
            color_discrete_sequence= ColPalDuo, 
            opacity=opacity,
            hover_data=hover_data,
            custom_data=['index'],
            template='simple_white',
            render_mode = 'webgl')
        fig1.update_traces(marker=dict(size=pointsize))
        fig1.update(layout_coloraxis_showscale=False)

        if highlight:
            highlighted = matrix[matrix[node_name].isin(highlight)]
            fig3 = px.scatter(
                highlighted,
                x=x,
                y=y,
                hover_name=node_name,
                color_discrete_sequence=ColPalDuo,
                hover_data=hover_data,
                opacity=1,
                custom_data=['index'],
                template='simple_white',
                text=highlighted[node_name],
                render_mode = 'webgl'
                )
            fig3.update_traces(marker=dict(color=highlight_color, size=14, symbol = "circle-open"))
            fig3.update_traces(textposition="top center", textfont = dict(size=14, color = highlight_color))
            fig3.update(layout_coloraxis_showscale=False)
        else:
            fig3 = go.Figure()
        
        if categorical:
            fig = go.Figure(data=fig2.data + fig1.data + fig3.data)
        else:
            fig = go.Figure(data=fig1.data)

    #fig.update_xaxes(showticklabels=False, title_text=label_x, ticks="")
    fig.update_xaxes(ticks="", tickfont={"size":1}, showticklabels=True, title_text=label_x)
    fig.update_yaxes(ticks="", tickfont={"size":1}, showticklabels=True, title_text=label_y)

    fig.update_layout(
        template='simple_white',
        legend=dict(
            font=dict(size=14)
        )
    )

    return fig

def interaction_3D_umap(
        matrix, node_name, cluster, x='3D_umap_1', y='3D_umap_2', z='3D_umap_3', opacity=0.7,
        width=800, height=600, highlight=None, unlabelled_color='#D0D3D4',
        unlabelled_opacity=0.3, hover_data=None, unlabelled_hover=True, search=False,
        categorical=True, na_in_data=True, highlight_color="red", pointsize=6):

    matrix = matrix.copy()
    matrix.reset_index(inplace=True, drop=False)


    if node_name == 'None':
        node_name = None
    if 'umap' in x:
        label_x = 'UMAP 1'
        label_y = 'UMAP 2'
        label_z = 'UMAP 3'
    else:
        label_x = 'UMAP 1'
        label_y = 'UMAP 2'
        label_z = 'UMAP 3'

    if cluster == 'None':
        fig = px.scatter_3d(
            matrix,
            x=x,
            y=y,
            z=z,
            labels={
                x: label_x,
                y: label_y,
                z: label_z
            },
            hover_name=node_name,
            hover_data=hover_data,
            opacity=opacity,
            custom_data=['index'],
            template='simple_white')
        fig.update_traces(marker=dict(size=3.5))

    else:
        if na_in_data:
            labelling = matrix[cluster].isna()
            labelled = matrix[~labelling]
            if categorical:
                labelled[cluster] = labelled[cluster].astype(str)
            unlabelled = matrix[labelling]
            unlabelled[cluster] = 'unlabelled'
        else:
            labelled = matrix
            if categorical:
                labelled[cluster] = labelled[cluster].astype(str)
            unlabelled = matrix


        labelled.sort_values(by=cluster, inplace=True)

        fig2 = px.scatter_3d(
            unlabelled,
            x=x,
            y=y,
            z=z,
            labels={
                x: label_x,
                y: label_y,
                z: label_z
            },
            color=cluster,
            hover_name=node_name,
            opacity=unlabelled_opacity,
            hover_data=hover_data,
            custom_data=['index'],
            color_discrete_sequence=[unlabelled_color],
            template='simple_white')
        if not unlabelled_hover:
            fig2.update_traces(hoverinfo='skip', hovertemplate=None)
        fig2.update_traces(marker=dict(size=pointsize-3))

        #ColPalDuo = px.colors.qualitative.D3 + [Lp[1] , Lp[4] , Dp[5] , Lp[17] , Dp[14] , Dp[7] , Lp[22], Ap[24] , Ap[10], Ap[0], Lp[0], Dp[9], Dp[6] ]
        ColPalDuo = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10 + px.colors.qualitative.T10
        to_remove = ["#D62728", "#EF553B"]
        ColPalDuo = [x for x in ColPalDuo if x not in to_remove]

        fig1 = px.scatter_3d(
            labelled,
            x=x,
            y=y,
            z=z,
            labels={
                x: label_x,
                y: label_y,
                z: label_z
            },
            hover_name=node_name,
            color=cluster,
            #color_continuous_scale=px.colors.cyclical.mygbm[: -1], # small palette (Kibeom)
            #color_discrete_sequence=px.colors.qualitative.Dark24, # Duo: expand the color palette # for more options, see https://plotly.com/python/discrete-color/
            color_discrete_sequence=ColPalDuo, # Duo: expand the color palette # for more options, see https://plotly.com/python/discrete-color/
            opacity=opacity,
            hover_data=hover_data,
            custom_data=['index'],
            template='simple_white')
        fig1.update_traces(marker=dict(size=pointsize-3))
        fig1.update(layout_coloraxis_showscale=False)

        if highlight:
            highlighted = matrix[matrix[node_name].isin(highlight)]
            fig3 = px.scatter_3d(
                highlighted,
                x=x,
                y=y,
                z=z,
                hover_name=node_name,
                color_discrete_sequence=ColPalDuo,
                hover_data=hover_data,
                opacity=1,
                custom_data=['index'],
                template='simple_white',
                text=highlighted[node_name]
                )
            fig3.update_traces(marker=dict(color=highlight_color, size=10, symbol = "circle-open"))
            fig3.update_traces(textposition="top center", textfont = dict(size=14, color = highlight_color))
            fig3.update(layout_coloraxis_showscale=False)
        else:
            fig3 = go.Figure()

        if categorical:
            fig = go.Figure(data=fig2.data + fig1.data + fig3.data)
        else:
            fig = go.Figure(data=fig1.data)

    #fig.update_xaxes(showticklabels=False, title_text=label_x, ticks="")
    #fig.update_yaxes(showticklabels=False, title_text=label_y, ticks="")
    fig.update_layout(scene = dict(
                    xaxis_title=label_x,
                    yaxis_title=label_y,
                    zaxis_title=label_z)
                    )
    fig.update_layout(scene = dict(
                    xaxis = dict(
                        ticks="", tickfont={"size":1}),
                    yaxis = dict(
                        ticks="", tickfont={"size":1},),
                    zaxis = dict(
                        ticks="", tickfont={"size":1},))
                    )
    fig.update_layout(
        template='simple_white',
        legend=dict(
            font=dict(size=14)
        ),
        uniformtext_minsize=14,
        uniformtext_mode='hide'
    )
    
    return fig

def confidence_umap(
        matrix, node_name, colname, x='umap_1', y='umap_2', opacity=0.95,
        width=800, height=600, highlight=None, full_color='#D0D3D4',
        full_opacity=0.95, hover_data=None, unlabelled_hover=True, search=False,
        categorical=True, na_in_data=True, full_dot_size=6, partial_dot_size=6):

    matrix = matrix.copy()
    matrix.reset_index(inplace=True, drop=False)

    full_conf = matrix[matrix[colname] == 1]
    full_conf[colname] = full_conf[colname].apply(lambda x: str(int(x)))
    partial_conf = matrix[(matrix[colname] < 1) & (matrix[colname] >= 0)]

    x='umap_1'
    y='umap_2'
    label_x = 'UMAP 1'
    label_y = 'UMAP 2'

    fig2 = px.scatter(
        full_conf,
        x=x,
        y=y,
        labels={
            x: label_x,
            y: label_y
        },
        color=colname,
        hover_name=node_name,
        opacity=full_opacity,
        hover_data=hover_data,
        custom_data=['index'],
        color_discrete_sequence=[full_color],
        template='simple_white')
    fig2.update_traces(marker=dict(size=full_dot_size))

    fig1 = px.scatter(
        partial_conf,
        x=x,
        y=y,
        labels={
            x: label_x,
            y: label_y
        },
        hover_name=node_name,
        color=colname,
        #color_continuous_scale=px.colors.cyclical.mygbm[: -1], # small palette (Kibeom)
        #color_discrete_sequence=px.colors.qualitative.Dark24, # Duo: expand the color palette # for more options, see https://plotly.com/python/discrete-color/
        opacity=opacity,
        hover_data=hover_data,
        custom_data=['index'],
        template='simple_white')
    fig1.update_traces(marker=dict(size=partial_dot_size))
    fig1.update(layout_coloraxis_showscale=False)

    fig = go.Figure(data=fig2.data + fig1.data)
    fig.update_xaxes(showticklabels=False, title_text=label_x, ticks="")
    fig.update_yaxes(showticklabels=False, title_text=label_y, ticks="")
    fig.update_layout(
        template='simple_white',
        legend=dict(
            font=dict(size=14)
        )
    )

    return fig