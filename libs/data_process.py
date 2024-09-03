import numpy as np
import pandas as pd
import networkx as nx
import re
from sklearn.model_selection import train_test_split

def keep_retweet_interactions(df, keep="original", t=2):
    '''
    Returns a dataframe with interactions corresponding to the type of network
    Input:
        df: dataframe for retweet network
        keep: original, active, strong
        t: keep interactions equal or greater than t
    '''
    df = df.loc[df["ToUser"] != df["FromUser"]].reset_index(drop=True) # remove self-loops
    if keep == "active":
        # Users with less than t retweets
        remove_users = (
            df
            .groupby("FromUser")
            ["count"]
            .sum()
            .reset_index(name="num_retweets")
            .loc[lambda df: df['num_retweets'] < t]
        )
        # Remove users with less than t retweets 
        df = (
            df
            .loc[~df['FromUser'].isin(remove_users['FromUser'])]
            .reset_index(drop=True)
        )
    elif keep == "strong":
        df = df.groupby(["FromUser", "ToUser"])["count"].sum().reset_index(name="count") # compute weights
        df = df.loc[~(df['count'] < t)].reset_index(drop=True) 
        
    df[["FromUser", "ToUser"]] = np.sort(df[["FromUser", "ToUser"]].values, axis=1) 
    df = df.groupby(["FromUser", "ToUser"])["count"].sum().reset_index(name="weight") # compute weight
    return df


def get_gcc_network(df, verbose=False):
    '''
    Returns Giant Connected Component of Network
    Input:
        - df: retweet network in tabular format
        - verbose: print network size if True
    '''
    
    G = nx.from_pandas_edgelist(df, "FromUser", "ToUser", edge_attr=["weight"],
                                create_using=nx.Graph()) # create networkX graph
    if verbose:
        print("Network size:")
        print(G)
        print()
    
    # Extract the giant connected component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    Gcc = G.subgraph(Gcc[0])
    if verbose:
        print("GCC size:")
        print(Gcc)
    return Gcc   


def generate_topic_label_per_user(df):
    '''
    Returns a list of users and a list of topic labels
    '''
    df = (
        df
        .groupby(['nodeUserID', 'topic']) 
        ['nodeID']
        .nunique() # compute number of tweets per topic
        .reset_index()
        .sort_values(['nodeUserID', 'nodeID'], ascending=[True, False])
        .drop_duplicates('nodeUserID', keep='first') # keep topic with highest number of tweets for each user
    )
    users, labels = list(df['nodeUserID'].values), list(df['topic'].values)
    return users, labels 


def generate_train_test_split(data, labels, size_ratio=None, rs=42):
    '''
    Return splits for users
    '''
    
    train, test, train_labels, test_labels = train_test_split(data, 
                                                              labels, 
                                                              test_size=1-size_ratio, 
                                                              stratify=labels, 
                                                              random_state=rs, 
                                                              shuffle=True)
    return train, test, train_labels, test_labels


def compute_topic_probability(df, user_list):
    '''
    Returns a dictionary where key is topic and value is the fraction of users
    per topic
    Input:
        - df: dataframe of tweets
        - user_list: list of users to consider
    '''
    df = df.loc[df['nodeUserID'].isin(user_list)].reset_index(drop=True)
    topic_weights = (
        df
        .groupby("topic")
        ["nodeUserID"]
        .nunique()
        .reset_index(name="count")
        .assign(weight=lambda df: df["count"]/df["count"].sum())
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    return dict(zip(topic_weights.topic, topic_weights.weight))
    

# Helper Functions to Preprocess Text
def clean_text(x):
    x = str(x)
    x = re.sub("\B@\w+", "", x) # remove mentions
    x = re.sub("(?:\\@|http?\\://|https?\\://|www)\\S+", "", x) # remove URLs
    
    x = x.lower()
    x = x.replace(' ','')
    x = re.sub('[^0-9a-zA-Z]+', '', x)
    return x

def just_text(x):
    x = re.sub("(?:\\@|http?\\://|https?\\://|www)\\S+", "", x)
    return x