import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import umap

colors = ["#a6cee3",
"#1f78b4",
"#b2df8a",
"#33a02c",
"#fb9a99",
"#e31a1c",
"#fdbf6f",
"#ff7f00",
"#cab2d6",
"#6a3d9a",
"#ffff99",
"#b15928",
"black"]


def plot_embeddings(df):
    global colors
    sns.set_context("talk",font_scale=0.8)

    data = df[[c for c in df.columns if 'emb' in c]].values
    # UMAP
    um = umap.UMAP(metric='cosine', random_state=42)
    um2d = um.fit_transform(data)

    df["umap0"] = um2d[:,0]
    df["umap1"] = um2d[:,1]
    df['label'] = df['topic'].astype(str) + ' - ' + df['stance'].astype(str)

    fig,ax = plt.subplots(figsize=(10,6))
    count = 0
    sample = df.copy()
    sample["count"] = sample.groupby('label')["tweet_id"].transform('count')
    sample = sample.sort_values('count', ascending=False).reset_index(drop=True)

    for g,grp in sample.groupby('label', sort=True):
        grp.plot(x='umap0',y='umap1',label=g,kind='scatter',ax=ax,color=colors[count])
        count += 1
    plt.legend(fontsize=12, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_line(df, params=None):
    n_list = np.sort(df['N'].unique())
    
    x = params["x"]
    y = params["y"]
    hue_col = params["hue_col"]
    
    df["label"] = df[hue_col].map(params["label_map"])
    
    fig, ax = plt.subplots(figsize=params["figsize"])
    
    g = sns.lineplot(x=x, y=y, data=df, 
                     hue='label', palette=params["color_map"],
                     marker='o', hue_order=params["hue_order"],
                     linewidth=3,
                     markersize=10,
                     ax=ax)
    legend_handles, _= g.get_legend_handles_labels()
    g.legend(bbox_to_anchor=(1,1), fontsize=20)
    ax.set_xlabel(x, fontsize=20)
    ax.set_ylabel(y, fontsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.xticks(n_list)
    plt.tight_layout()
    myfig = plt.gcf()
    plt.show()
    return myfig


def plot_scatter(df, params=None, savepath=None):
    
    x = params["x"]
    y = params["y"]
    hue_col = params["hue_col"]
    x_base = params["vertical_bound"]
    y_base = params["horizontal_bound"]
    
    df["label"] = df[hue_col].map(params["label_map"])
    min_lim = math.floor(min(min(df[x]), min(df[y])) * 10)/10
    
    fig, ax = plt.subplots(figsize=params["figsize"])
    
    g = sns.scatterplot(x=x, y=y, data=df, 
                    hue='label', palette=params["color_map"],
                    s=params["markersize"], hue_order=params["hue_order"],
                    ax=ax)
    legend_handles, _= g.get_legend_handles_labels()
    g.legend(bbox_to_anchor=(1,1), fontsize=14)

    ax.set_xlim(min_lim, 1)
    ax.set_ylim(min_lim, 1)
    ax.set_xlabel('TopicNN', fontsize=15)
    ax.set_ylabel('ViewpointNN', fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    
    if y_base:
        ax.axhline(y=y_base, color='black', linestyle='--')
    if x_base:
        ax.axvline(x=x_base, color='black', linestyle='--')
    
    plt.tight_layout()
    myfig = plt.gcf()
    plt.show()
    return myfig