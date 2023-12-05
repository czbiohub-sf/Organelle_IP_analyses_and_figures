import matplotlib.pyplot as plt
import matplotlib
from numbers import Number
import numpy as np
import pandas as pd
import plotly.offline
from plotly import graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import math


def simple_volcano(v_df, bait, fcd, width=None, height=None):
    """plot the volcano plot of a given bait"""
    v_df = v_df.copy()
    v_df.set_index(('metadata', 'Gene names'), inplace=True)

    # Specify the bait column
    bait_vals = v_df[bait]
    bait_vals['thresh'] = bait_vals['enrichment'].apply(calc_thresh,
        args=[fcd[0], fcd[1]])
    bait_vals['hits'] = np.where((bait_vals['pvals'] > bait_vals['thresh']), True, False)


    hits = bait_vals[bait_vals['hits']]
    print("Number of Significant Hits: " + str(hits.shape[0]))
    no_hits = bait_vals[~bait_vals['hits']]

    xmax = hits['enrichment'].max() + 3
    if hits.shape[0] > 0:
        ymax = hits['pvals'].max() + 4
    else:
        ymax = 30

    # FCD plot calculation
    x1 = np.array(list(np.linspace(-12, -1 * fcd[1] - 0.001, 200))
        + list(np.linspace(fcd[1] + 0.001, 12, 200)))
    y1 = fcd[0] / (abs(x1) - fcd[1])
    # x2 = np.array(list(np.linspace(-12, -1 * fcd2[1] - 0.001, 200))
    #     + list(np.linspace(fcd2[1] + 0.001, 12, 200)))
    # y2 = fcd2[0] / (abs(x2) - fcd2[1])


    # Figure Generation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hits['enrichment'], y=hits['pvals'],
        mode='markers+text', text=hits.index.tolist(), textposition='bottom right',
        opacity=0.6, marker=dict(size=10)))
    fig.update_traces(mode='markers+text', marker_line_width=2)
    fig.add_trace(go.Scatter(x=no_hits['enrichment'], y=no_hits['pvals'],
        mode='markers', text=no_hits.index.tolist(), opacity=0.4, marker=dict(size=8)))

    fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines',
        line=dict(color='royalblue', dash='dash')))
    # fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines',
    #     line=dict(color='firebrick', dash='dash')))

    fig.update_layout(
        title={'text': bait,
            'x': 0.5,
            'y': 0.95},
            xaxis_title='Enrichment (log2)',
            yaxis_title='P value (-log10)',
            showlegend=False,
            margin={'l': 30, 'r': 30, 'b': 20, 't': 40})
    if width:
        fig.update_layout(
            width=width,
            height=height,
        )
    fig.update_xaxes(range=[-1 * xmax, xmax])
    fig.update_yaxes(range=[-1, ymax])

    return fig


def volcano_plot(v_df, bait, plate, marker='prey', marker_mode=True, width=None,
        height=None, experiment=False, fcd=True, color=None, unlabelled_hover=True,
        fcd_curve=True, search=False):
    # initiate dfs
    sel_df = v_df.copy()


    if experiment:
        bait_vals = sel_df[(sel_df['target'] == bait) & (sel_df['experiment'] == plate)]
    else:
        bait_vals = sel_df[sel_df['target'] == bait]

    if color:
        bait_vals[color] = bait_vals[color].fillna('Unlabelled')

    bait_vals.reset_index(drop=False, inplace=True)


    # calculations for x axis min, max parameters
    xmax = bait_vals['enrichment'].max() + bait_vals['enrichment'].max() * 0.1

    ymax = bait_vals['pvals'].max() + bait_vals['pvals'].max() * 0.1


    if fcd:
        bait_vals['interaction'] = bait_vals['interaction'].astype(bool)

        hits = bait_vals[bait_vals['interaction']]
        # print("Number of Significant Hits: " + str(hits.shape[0]))
        no_hits = bait_vals[~bait_vals['interaction']]

        if marker_mode:
            text = hits[marker].to_list()
        else:
            text = None


        # add non-significant hits
        fig2 = px.scatter(no_hits, x='enrichment', y='pvals',
            hover_name=marker, custom_data=['index'],
            opacity=0.4, color=color)
        fig2.update_traces(marker_size=8, marker_symbol='circle-open')
        if not color:
            fig2.update_traces(marker_color='#70aee0')
        if not unlabelled_hover:
            fig2.update_traces(hoverinfo='skip', hovertemplate=None)

        if search:
            fig = fig2
            fig.update_traces(name='Unlabelled')
            if marker_mode:
                search_marker = 'markers+text'
            else:
                search_marker = 'markers'
            fig.add_trace(
                go.Scattergl(x=hits['enrichment'], y=hits['pvals'], mode=search_marker,
                    name='search', hoverinfo='text', hovertext=hits[marker], text=text,
                    customdata=hits['index'].apply(lambda x: [x]), textposition='bottom right',
                    opacity=0.6, marker=dict(size=10, opacity=0.9, line_width=1, color='#EF553B'))
            )

        else:


            # add significant hits
            fig1 = px.scatter(hits, x='enrichment', y='pvals',
                hover_name=marker, text=text, custom_data=['index'],
                opacity=0.6, color=color)
            fig1.update_traces(marker_size=10, marker_line_width=1,
                textposition='bottom right')
            if not color:
                fig1.update_traces(marker_color='#EF553B')


            fig = go.Figure(data=fig2.data + fig1.data)

            # FCD curve calculation
            if fcd_curve:
                fcd1 = bait_vals.iloc[0]['fdr']

                x1 = np.array(list(np.linspace(-12, -1 * fcd1[1] - 0.001, 200))
                    + list(np.linspace(fcd1[1] + 0.001, 12, 200)))
                y1 = fcd1[0] / (abs(x1) - fcd1[1])

                fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='fcd_line',
                    line=dict(color='DarkSlateGrey', dash='dash')))





    else:

        fig = px.scatter(bait_vals, x='enrichment', y='pvals',
            hover_name=marker, custom_data=['index'],
            opacity=0.6, color=color)
        fig.update_traces(marker_size=8)
        if not color:
            fig.update_traces(marker_color='#70aee0')


    # axis customization
    fig.update_xaxes(title_text='Enrichment (log2)',
        range=[-1 * xmax, xmax])
    fig.update_yaxes(title_text='p-value (-log10)',
        range=[-1, ymax])
    for trace in fig['data']:
        if trace['name'] == 'fcd_line' or trace['name'] == 'search':
            trace['showlegend'] = False
    fig.update_layout(
        template='ggplot2',
        xaxis_title='Enrichment (log2)',
        yaxis_title='P value (-log10)',
        showlegend=True,
        margin={'l': 30, 'r': 30, 'b': 20, 't': 40})
    if width:
        fig.update_layout(
            width=width,
            height=height,
        )

    return fig


def comparison_volcano(v_df, v2_df, bait, fcd, fcd2):
    """plot volcano plots from two analyses for qualitative comparisons"""

    # initiate dfs
    v_df = v_df.copy()
    v2_df = v2_df.copy()
    v_dfs = [v_df, v2_df]
    fcds = [fcd, fcd2]

    # start a subplot
    fig = make_subplots(rows=1, cols=2)
    for i in [1, 2]:
        bait_vals = v_dfs[i-1][bait]
        hits = bait_vals[bait_vals['hits']]
        print(str(i) + "st Analysis: Number of Significant Hits: "
            + str(hits.shape[0]))
        no_hits = bait_vals[~bait_vals['hits']]

        # calculations for x axis min, max parameters
        xmax = hits['enrichment'].max() + 3
        if hits.shape[0] > 0:
            ymax = hits['pvals'].max() + 4
        else:
            ymax = 30

        # calculation for FCD threshold
        x1 = np.array(list(np.linspace(-12, -1 * fcds[i-1][1] - 0.001, 200))
            + list(np.linspace(fcds[i-1][1] + 0.001, 12, 200)))
        y1 = fcds[i-1][0] / (abs(x1) - fcds[i-1][1])

        # add significant hits
        fig.add_trace(go.Scatter(x=hits['enrichment'], y=hits['pvals'],
            mode='markers+text', text=hits.index.tolist(), textposition='bottom right',
            opacity=0.6, marker=dict(size=10, line=dict(width=2))), row=1, col=i)

        # add non-significant hits
        fig.add_trace(go.Scatter(x=no_hits['enrichment'], y=no_hits['pvals'],
            mode='markers', text=no_hits.index.tolist(), opacity=0.4,
            marker=dict(size=8)), row=1, col=i)

        fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines',
            line=dict(color='royalblue', dash='dash')), row=1, col=i)

        # axis customization
        fig.update_xaxes(title_text='Enrichment (log2)', row=1, col=i,
            range=[-1 * xmax, xmax])
        fig.update_yaxes(title_text='p-value (-log10)', row=1, col=i,
            range=[-1, ymax])

    # layout
    fig.update_layout(
        # width=1000,
        # height=600,
        width=800,
        height=400,
        title={'text': bait,
            'x': 0.5,
            'y': 0.98},
            showlegend=False,
            margin={'l': 30, 'r': 30, 'b': 20, 't': 40})
    fig.show()





def mult_volcano(v_df, baits):
    """plot the volcano plot of a given bait"""
    v_df = v_df.copy()
    fig = go.Figure()
    g_xmax = 0
    g_ymax = 0
    for bait in baits:
        bait_vals = v_df[bait]
        hits = bait_vals[bait_vals['hits']]
        print("Number of Significant Hits: " + str(hits.shape[0]))
        no_hits = bait_vals[~bait_vals['hits']]

        xmax = hits['enrichment'].max() + 3
        if hits.shape[0] > 0:
            ymax = hits['pvals'].max() + 4
        else:
            ymax = 30
        if xmax > g_xmax:
            g_xmax = xmax
        if ymax > g_ymax:
            g_ymax = ymax

        fig.add_trace(go.Scatter(x=hits['enrichment'], y=hits['pvals'],
            mode='markers', text=hits.index.tolist(), textposition='bottom right',
            opacity=0.4, marker=dict(size=10), name=bait))
        fig.update_traces(mode='markers', marker_line_width=2)
        fig.add_trace(go.Scatter(x=no_hits['enrichment'], y=no_hits['pvals'],
            mode='markers', opacity=0.4, marker=dict(size=8)))

    x1 = np.array(list(np.linspace(-8, -1.750001, 100)) + list(np.linspace(1.750001,
        8, 100)))
    y1 = 3.65 / (abs(x1)-1.75)
    x2 = np.array(list(np.linspace(-8, -0.9001, 100)) + list(np.linspace(0.90001,
        8, 100)))
    y2 = 2.9 / (abs(x2)-0.9)


    fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines',
        line=dict(color='royalblue', dash='dash')))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines',
        line=dict(color='firebrick', dash='dash')))

    fig.update_layout(
        title={'text': bait,
            'x': 0.5,
            'y': 0.95},
        xaxis_title='Enrichment (log2)',
        yaxis_title='P value (-log10)',
        showlegend=False,
        margin={'l': 30, 'r': 30, 'b': 20, 't': 40})
    fig.update_xaxes(range=[-1 * g_xmax, g_xmax])
    fig.update_yaxes(range=[-1, g_ymax])
    fig.show()

# def calc_thresh(enrich, fc_var1, fc_var2):
#     """simple function to get FCD thresh to recognize hits"""
#     if enrich < fc_var2:
#         return np.inf
#     else:
#         return fc_var1 / (abs(enrich) - fc_var2)


def two_fdrs(pval_df, fdr1, fdr2):
    """ compute 1% FDR and 5% FDR """

    pval_df = pval_df.copy()

    # get a list of baits
    baits = list(set([x[0] for x in list(pval_df) if x[0] != 'gene_names']))


    # Find hits for FDR1 and FDR2
    for bait in baits:
        pval = pval_df[bait]['pvals']
        enrichment = pval_df[bait]['enrichment']
        # 1% thresh

        first_thresh = enrichment.apply(calc_thresh,
            args=[fdr1[0], fdr1[1]])

        # 5% thresh
        second_thresh = enrichment.apply(calc_thresh,
            args=[fdr2[0], fdr2[1]])

        pval_df[(bait, 'hits')] = np.where(
            (pval > first_thresh), True, False)

        pval_df[(bait, 'minor_hits')] = np.where(
            ((pval < first_thresh) & (pval > second_thresh)), True, False)

    pval_df.sort_index(axis=1, inplace=True)
    return pval_df


def all_hits_two_fdrs(pval_df, fdr1, fdr2):
    """ compute 1% fdr and 5 % FDR on all_hits table """
    pval_df = pval_df.copy()

    pval = pval_df['pvals']
    enrichment = pval_df['enrichment']
    first_thresh = enrichment.apply(calc_thresh,
            args=[fdr1[0], fdr1[1]])

    # 5% thresh
    second_thresh = enrichment.apply(calc_thresh,
        args=[fdr2[0], fdr2[1]])

    pval_df['hits'] = np.where(
        (pval > first_thresh), True, False)

    pval_df['minor_hits'] = np.where(
        ((pval < first_thresh) & (pval > second_thresh)), True, False)
    # pval_df = pval_df[(pval_df['hits']) | (pval_df['minor_hits'])]
    pval_df.reset_index(inplace=True, drop=True)
    return pval_df


def comparison_volcano_temp(v_df, bait, width=800, height=400, show=False):
    """plot volcano plots from two analyses for qualitative comparisons"""

    # initiate dfs
    sel_df = v_df.copy()
    sel_df = v_df.set_index('prey')
    # plates = list(set(sel_df[
    #     sel_df['target'].apply(lambda x: x.split('_')[0]) == bait]['plate'].to_list()))

    plates = list(set(sel_df[
        sel_df['target'] == bait]['plate'].to_list()))

    plates.sort()
    num_plates = len(plates)

    subplot_titles = [bait + ' ' + plate for plate in plates]

    # start a subplot
    fig = make_subplots(rows=math.ceil(num_plates/2), cols=1,
        subplot_titles=subplot_titles, vertical_spacing=0.125)
    # layout
    fig.update_layout(
        # width=1000,
        # height=600,
        width=width,
        height=height * math.ceil(num_plates/2),
        showlegend=False,
        margin={'l': 30, 'r': 30, 'b': 20, 't': 40})

    hit_counts = []
    minor_hit_counts = []
    for i, plate in enumerate(plates):
        # bait_vals = sel_df[
        #     (sel_df['target'].apply(lambda x: x.split('_')[0]) == bait) & (sel_df['plate'] == plate)]

        bait_vals = sel_df[
            (sel_df['target'] == bait) & (sel_df['plate'] == plate)]


        hits = bait_vals[bait_vals['hits']]
        hit_counts.append(hits.shape[0])
        # print("Number of Significant Hits: " + str(hits.shape[0]))

        minor_hits = bait_vals[bait_vals['minor_hits']]
        # print("Number of Minor Hits: " + str(minor_hits.shape[0]))
        minor_hit_counts.append(minor_hits.shape[0])

        no_hits = bait_vals[(~bait_vals['hits']) | (~bait_vals['minor_hits'])]


        # calculations for x axis min, max parameters
        xmax = hits['enrichment'].max() + 3
        if hits.shape[0] > 0:
            ymax = hits['pvals'].max() + 4
        else:
            ymax = 30

        # FCD plot calculation
        fcd1 = bait_vals.iloc[0]['fdr1']
        fcd2 = bait_vals.iloc[0]['fdr5']


        x1 = np.array(list(np.linspace(-12, -1 * fcd1[1] - 0.001, 200))
            + list(np.linspace(fcd1[1] + 0.001, 12, 200)))
        y1 = fcd1[0] / (abs(x1) - fcd1[1])
        x2 = np.array(list(np.linspace(-12, -1 * fcd2[1] - 0.001, 200))
            + list(np.linspace(fcd2[1] + 0.001, 12, 200)))
        y2 = fcd2[0] / (abs(x2) - fcd2[1])


        row_num = math.ceil((i + 1) / 2)
        if (i + 1) % 2 == 1:
            col_num = 1
        else:
            col_num = 2
        # add significant hits
        fig.add_trace(go.Scatter(x=hits['enrichment'], y=hits['pvals'],
            mode='markers+text', text=hits.index.tolist(), textposition='bottom right',
            opacity=0.6, marker=dict(size=10, line=dict(width=2)),
            name="Significant hits: " + str(hits.shape[0])), row=row_num, col=col_num)

        # add minor hits
        fig.add_trace(go.Scatter(x=minor_hits['enrichment'], y=minor_hits['pvals'],
            mode='markers+text', text=minor_hits.index.tolist(), textposition='bottom right',
            opacity=0.6, marker=dict(size=10, color='firebrick'),
            name="Minor hits: " + str(minor_hits.shape[0])), row=row_num, col=col_num)

        # add non-significant hits
        fig.add_trace(go.Scatter(x=no_hits['enrichment'], y=no_hits['pvals'],
            mode='markers', text=no_hits.index.tolist(), opacity=0.4,
            marker=dict(size=8)), row=row_num, col=col_num)

        fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines',
            line=dict(color='royalblue', dash='dash')), row=row_num, col=col_num)
        fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines',
            line=dict(color='firebrick', dash='dash')), row=row_num, col=col_num)
        # axis customization
        fig.update_xaxes(title_text='Enrichment (log2)', row=row_num, col=col_num,
            range=[-1 * xmax, xmax * 1.2])
        fig.update_yaxes(title_text='p-value (-log10)', row=row_num, col=col_num,
            range=[-1, ymax])
    if show:
        fig.show()
    return fig


def calc_thresh(enrich, fc_var1, fc_var2):
    """simple function to get FCD thresh to recognize hits"""

    if enrich < fc_var2:
        return np.inf

    elif (enrich == 0) & (fc_var2 == 0):
        return np.inf

    else:
        return fc_var1 / (abs(enrich) - fc_var2)
    # fig.write_image('ignore/old_pickles/1201/' + bait +'.pdf')

    # # fig.show()
    # counts = pd.DataFrame()
    # counts['plate'] = plates
    # counts['major_hits'] = hit_counts
    # counts['minor_hits'] = minor_hit_counts
    # # return counts
    # # fig.write_image('/ignore/old_pickles/1201/' + bait +'.png')
    # return fig
