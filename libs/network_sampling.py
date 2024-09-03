import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
import math


def numpy_combinations(x):
    ''' 
    Returns all pairwise combinations from a list
    
    x - numpy array
    '''
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]


def get_random_pairs(users, size=1000):
    ''' 
    Returns random pairs of a given size

    users - numpy array of users to consider for pairwise combinations
    size - number of pairs to sample
    '''
    pairs = numpy_combinations(users)
    idx = np.random.randint(len(pairs), size=int(size))
    return pairs[idx]


def get_random_tweet_pair(source, df=None):
    '''
    Returns a random tweet for source
    '''
    tweet_source = df.loc[df['nodeUserID']==source].sample(1)['nodeID'].item()
    return tweet_source


def one_hop_sampling(G, seed):
    ''' 
    Returns an edgelist of one-hop neighbors for a given seed
    Inputs:
        - G: networkx graph object
        - seed: seed user
    '''
    edgelist = []
    hops=[]
   
    neighbors = list(G.neighbors(seed))
    for n in neighbors:
        if (G.nodes[n]["node_type"] == 1): # only append to edgelist engaged users (i.e., node_type is 1)
            edgelist.append((seed, n))
            hops.append(1)
    return edgelist, hops


def two_hop_sampling(G, seed, max_neighbors=10):
    ''' 
    Returns an edgelist of two-hop neighbors for a given seed
    Inputs:
        - G: networkx graph object
        - seed: seed user
        - max_neighbors: maximum number of neighbors to explore
    '''
    edgelist = []
    hops = []
    
    visited = {seed}
    neighbors = list(G.neighbors(seed))
    
    # add one-hop neighbors to visited set
    visited = visited | set(neighbors)
    
    # Randomly sample N neighbors
    if len(neighbors) > max_neighbors:
        neighbors = np.random.choice(neighbors, size=max_neighbors, replace=False)
    
    for node in neighbors:
        # Explore neighbors two hops away from seed
        twohops = list(G.neighbors(node))
        for n in twohops:
            if (G.nodes[n]["node_type"] == 1) and (n not in visited):
                # add to edge list only if not visited and if it's a poster user
                edgelist.append((seed, n))
                hops.append(2)
        visited = visited | set(twohops) # update visited list
    return edgelist, hops


def sample_close_pairs(G, 
                       g=None, 
                       users=[], 
                       non_posters=[], 
                       n_target=0, 
                       max_neighbors=0, 
                       frac_one_hop=0.5, 
                       max_onehop_pairs=0,
                       max_two_hop_pairs=0):
    ''' 
    Returns a dataframe of closely connected pairs with columns: source, target, sp (shortest path lenght)
    Inputs:
        - G: networkx graph object for social media interactions
        - g: a graph object to track sampled pairs
        - users: list of users in the current data split
        - non_posters: list of users not in the current data split
        - n_target: number of close pairs to sample
        - max_neighbors: maximum number of neighbors to explore
        - frac_one_hop: fraction of pairs from one-hop neighborhood
        - max_onehop_pairs: maximum number of pairs to sample per seed user. Lower number results in more diverse set of users
        - max_twohop_pairs: maximum number of pairs to sample per seed user
    '''
    edges_list = []
    dists = [] 
    
    users_subgraph = users + non_posters # users to induce subgraph on
    
    # create a dictionary for attributes (1 means user in split, 0 means user not in split)
    map_node_type = pd.DataFrame(users_subgraph, columns=["user"])
    map_node_type["type"] = np.where(map_node_type["user"].isin(users), 1, 0)
    map_node_type = dict(zip(map_node_type["user"], map_node_type["type"]))
    nx.set_node_attributes(G, values=map_node_type, name="node_type")
    
    for seed in tqdm(users): # Append one-hop pairs until we reach our target 
        edges, hops = one_hop_sampling(G, seed=seed)
        ran_idx = np.random.choice(range(len(edges)), size=min(len(edges), max_onehop_pairs), replace=False) # Take a random sample of edges
        edges = [edges[i] for i in ran_idx]
        hops = [hops[i] for i in ran_idx]
        g.add_edges_from(edges) # update number of edges added
        edges_list.extend(edges)
        dists.extend(hops)

        if g.number_of_edges() >= (n_target*frac_one_hop): # Break if we hit the desired number of one-hop edges
            break

    random.shuffle(users) # Shuffle seed users
    for seed in tqdm(users): # Append two-hop pairs until we reach our target
        edges, hops = two_hop_sampling(G, seed=seed, max_neighbors=max_neighbors)
        ran_idx = np.random.choice(range(len(edges)), size=min(len(edges), max_two_hop_pairs), replace=False)
        edges = [edges[i] for i in ran_idx]
        hops = [hops[i] for i in ran_idx]
        g.add_edges_from(edges)
        edges_list.extend(edges)
        dists.extend(hops)

        if g.number_of_edges() >= n_target: # Break if we hit desired target of edges
            break

    # Build dataframe of close pairs        
    df = pd.DataFrame(edges_list, columns=["source", "target"])
    df["sp"] = dists
    df[["source","target"]] = np.sort(df[["source", "target"]].values, axis=1)
    df = df.drop_duplicates(["source","target"]).reset_index(drop=True)
    if len(df) > n_target: # Remove excess of samples, if any
        df = df.sample(n=int(n_target)).reset_index(drop=True)
    return df


def sample_distant_pairs(G, 
                         g=None,
                         users=[], 
                         n_target=0, 
                         n_users=0,
                         n_pairs=0, 
                         verbose=True):
    ''' 
    Returns a dataframe of distant pairs with columns: source, target, sp (shortest path lenght)

    Input:
        - G: networkx graph object for retweet network
        - g: a graph object to keep track of sampled pairs
        - users: list of users in data split
        - n_target: number of pairs to sample
        - n_users: number of users to draw randomly
        - n_pairs: number of pairs to sample in each draw
    '''
    edges = []
    num_target = n_target

    count_consecutive_zero_iter = 0 # number of consecutive iterations with no pairs added
    while (num_target > 0) and (count_consecutive_zero_iter != 10):
        count_pairs_in_iter = 0

        rand_users = np.random.choice(users, size=min(n_users,len(users)), replace=False)
        total_pairs = (len(rand_users)*(len(rand_users)-1))/2
        pairs = get_random_pairs(rand_users, size=min(n_pairs, total_pairs))
        
        for u,v in pairs: 
            if g.has_edge(u,v): # check if edge is not valid
                continue
            else:
                g.add_edge(u,v)
                edges.append((u,v))
                num_target -=1
                count_pairs_in_iter += 1
        if count_pairs_in_iter == 0:
            count_consecutive_zero_iter += 1
            print(f"Distant pairs added to candidate pool: {count_pairs_in_iter}...")
        else:
            count_consecutive_zero_iter = 0 # set the consecutive iterations to zero
    
    # Build dataframe of distant pairs
    df = pd.DataFrame(edges, columns=["source", "target"])
    df[["source","target"]] = np.sort(df[["source", "target"]].values, axis=1)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) > n_target:
        df = df.sample(n=int(n_target)).reset_index(drop=True)
    return df


def sample_self_loops(df, size=0):
    '''
    Helper function for generating pairs with 0 shortest path length
    
    Return a random sample of particular size. Or all pairs if len(pairs) < size
    '''
    df = df.sample(n=min(size, len(df))).reset_index(drop=True)
    df['source'] = df['nodeUserID']
    df['target'] = df["nodeUserID"]
    df['sp'] = 0
    df[['source_tweet', 'target_tweet']] = df["pairs"].to_list()
    df = df.drop(columns=['nodeUserID', 'nodeID', 'pairs'])
    return df


def filter_one_tweet_users(df):
    '''
    Returns a dataframe with users who have at least 2 tweets
    '''
    df = df.loc[df['nodeID'].str.len()>1].reset_index(drop=True)
    return df


def generate_self_loops(df, size=0):
    '''
    Returns a dataframe of pairs of users with 0 shortest path lenght
    '''
    return (df
            .groupby('nodeUserID')
            ['nodeID']
            .apply(list)
            .reset_index()
            .pipe(filter_one_tweet_users)
            .assign(
                pairs = lambda df: df.apply(lambda x: numpy_combinations(np.array(x["nodeID"])), axis=1)
            )
            .explode('pairs')
            .pipe(sample_self_loops, size=size) 
           )


def sample_pairs_per_topic(G, 
                           users=[], 
                           non_posters=[], 
                           n_target=0,
                           self_df=None,
                           max_neighbors=0, 
                           n_users=0,
                           n_pairs=0, 
                           frac_one_hop=0.5,
                           close_pairs_split=0.5,
                           max_onehop_pairs=0,
                           max_two_hop_pairs=0,
                           verbose=True):
    
    '''
    Returns a dataframe of close and distant pairs of users for a particular topic
    
    Inputs:
        - G: networkx graph of interactions
        - users: seed users of interest
        - non_posters: remaining users from the network who do not generate content
        - n_target: number of total pairs to sample for a particular topic
        - self_df: dataframe filtered by topic and by users of interest
        - max_neighbors: maximum number of two-hop neighbors to explore 
        - n_users: number of users to randomly sample for distant pairs
        - n_pairs: number of pairs to randomly sample for distant pairs
        - frac_one_hop: the fraction of pairs coming from one-hop neighborhoods
        - close_pairs_split: the desired fraction of pairs from close connections out of the total number of pairs
        - max_onehop_pairs: maximum number of pairs to sample per seed user. Lower number results in more diverse set of users
        - max_twohop_pairs: maximum number of pairs to sample per seed user
    '''
    g = nx.Graph() # Empty network to keep track of already sampled pairs
    
    total_target_close = math.ceil(n_target*close_pairs_split) # Total number of pairs from close connections (0, 1, and 2 hops)
    n_target_distant = n_target - total_target_close # Total number of pairs from distant connections
    n_target_close = math.floor(total_target_close*(1/3)) # Sample same fraction of 0, 1 and 2 hop pairs
    # Sample 0 hop pairs
    df_zero_hops = generate_self_loops(self_df, size=n_target_close)
    print(f'# 0-hop pairs:{len(df_zero_hops)}')

    # Update total target for close pairs
    total_target_close = total_target_close - len(df_zero_hops)
    n_target_close = total_target_close
    
    if len(users) <= n_target_close: # If there are less users than there are pairs to sample, then sample only one pair per user
        max_onehop_pairs = 1
        max_two_hop_pairs = 1
    
    # Sample one hop and two hop pairs
    df_close = sample_close_pairs(G, 
                                  g=g,
                                  users=users, 
                                  non_posters=non_posters,
                                  n_target=n_target_close, 
                                  max_neighbors=max_neighbors, 
                                  frac_one_hop=frac_one_hop, 
                                  max_onehop_pairs=max_onehop_pairs, 
                                  max_two_hop_pairs=max_two_hop_pairs)
    
    # Sample random tweets for pairs
    df_close['source_tweet'] = df_close['source'].apply(lambda x: get_random_tweet_pair(x, self_df))
    df_close['target_tweet'] = df_close['target'].apply(lambda x: get_random_tweet_pair(x, self_df))
    print(f'# 1-hop pairs:{len(df_close.loc[df_close["sp"]==1])}')
    print(f'# 2-hop pairs:{len(df_close.loc[df_close["sp"]==2])}')
    
    # Sample distant hops
    df_distant = sample_distant_pairs(G, g=g,
                                      users=users, 
                                      n_target=n_target_distant, 
                                      n_users=n_users, 
                                      n_pairs=n_pairs)
    
    # Sample random tweets for pairs
    df_distant['sp'] = -1 # add SPL placeholder
    df_distant['source_tweet'] = df_distant['source'].apply(lambda x: get_random_tweet_pair(x, self_df))
    df_distant['target_tweet'] = df_distant['target'].apply(lambda x: get_random_tweet_pair(x, self_df))
    print(f'# Distant pairs:{len(df_distant)}')
    
    return df_zero_hops, df_close, df_distant


def get_pairs_from_topics(G, df, users, network_users, params, n=0, non_network_users=None):
    
    '''
    Returns two dataframes: a dataframe of close pairs and a dataframe of distant pairs
    
    Inputs
        - G: network of retweet interactions
        - df: dataframe with tweets
        - users: list of users in the network to consider 
        - network_users: list of users in network
        - non_network_users: list of users who only generate content
        - params: dictionary of sampling parameters
        - n: total number of pairs to draw
    '''
    # Parameters for distant users sampling
    n_users = params["n_users"]
    n_pairs = params["n_pairs"]
    # Parameters for close users sampling
    max_onehop_pairs = params["max_onehop_pairs"]
    max_twohop_pairs = params["max_twohop_pairs"]
    max_neighbors = params["max_neighbors"]
    frac_one_hop = params["frac_one_hop"]
    close_pairs_split = params["close_pairs_split"]
    
    # Dictionary of topics weighted by number of users
    topic_weights = params["topic_weights"]
    
    if non_network_users==None:
        non_network_users=[]
        
    
    dfs_close=[] # store close pairs
    dfs_distant=[] # store distant pairs
    for topic, weight in topic_weights.items(): # iterate over topics
        print('Processing Topic:', topic)
        n_target = math.ceil(n*weight) # number of pairs to draw for topic
        
        # Filter dataframe by topic and users of interest
        topic_df = (df
                    .loc[(df['topic']==topic)&(df['nodeUserID'].isin(users+non_network_users))]
                    .reset_index(drop=True))
        topic_df = topic_df.sample(frac=1).reset_index(drop=True) # Shuffle entire data
        
        # Extract users who generate content and appear in the network
        users_intersect = list(set(users) & set(topic_df['nodeUserID']))
        # Extract remaining users
        other_network_users = list(set(network_users) - set(users_intersect))
        
        # df0 to store pairs with 0 hops; df12 for pairs with 1 and 2 hops; dfn are pairs with distant connections
        df0, df12, dfn = sample_pairs_per_topic(G, 
                                                users=users_intersect, 
                                                non_posters=other_network_users,
                                                n_target=n_target, 
                                                self_df=topic_df, 
                                                max_neighbors=max_neighbors, 
                                                frac_one_hop=frac_one_hop, 
                                                close_pairs_split=close_pairs_split, 
                                                max_onehop_pairs=max_onehop_pairs, 
                                                max_two_hop_pairs=max_twohop_pairs,
                                                n_users=n_users,
                                                n_pairs=n_pairs)

        dfs_close.append(pd.concat([df0,df12], ignore_index=True))
        dfs_distant.append(dfn)
        print('-'*100)
        print()
    
    dfs_close = pd.concat(dfs_close, ignore_index=True)
    dfs_distant = pd.concat(dfs_distant, ignore_index=True)
    return dfs_close, dfs_distant