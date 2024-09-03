import pandas as pd
import numpy as np 
from tqdm import tqdm
import math
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def compare_columns(row):
    '''
    Compares the value of one column with the other columns
    '''
    actual_class = row['actual_class']
    other_classes = row.drop('actual_class')
    comparisons = other_classes == actual_class
    return comparisons


def compute_knn_topic(df, n=1):
    '''
    Returns a dataframe using KNN classification for topics
    Inputs:
        - df: Dataframe with tweet ID, embeddings, and label information
        - n: number of nearest neighbors
    '''
    knn_df = df.copy()
    
    map_idx_to_stance = dict(zip(knn_df.index, knn_df.stance))
    map_idx_to_topic = dict(zip(knn_df.index, knn_df.topic))
    map_idx_to_tweet = dict(zip(knn_df.index, knn_df.tweet_id))
    
    X = knn_df[[c for c in knn_df.columns if 'emb' in c]].values
    knn = NearestNeighbors(n_neighbors=n+1, metric='euclidean', algorithm='brute').fit(X) # n+1 since the nearest neighbor of each point is itself
    distances, indices = knn.kneighbors(X)
    distances = distances[:, 1:].tolist() # remove distance to itself
    
    # Convert to dataframe
    indices_df = pd.DataFrame(indices, 
                              columns=["actual_class" if i == 0 else f"predict_{i}" for i in range(indices.shape[-1])])
    neighbors = indices_df.applymap(map_idx_to_tweet.get) # extract nearest neighbors tweet IDs
    classes = indices_df.applymap(map_idx_to_topic.get) # map tweet IDs to their topics
    
    predict_classes = classes.apply(compare_columns, axis=1) # returns a DF of True if same class otherwise False
    predict_classes.columns = [i.replace('predict', 'neighbor') for i in predict_classes.columns]
    classes = pd.concat([classes, predict_classes], axis=1)
    classes['majority_class'] = classes[[i for i in classes.columns if "neighbor" in i]].mode(axis=1) # majority vote
    return classes


def compute_knn_viewpoint(df, n=1):
    '''
    Returns a dataframe for predictions using KNN classification for viewpoints
    Inputs:
        - df: Dataframe with tweet ID, embeddings, and label information
        - n: number of nearest neighbors
    ''' 
    results = []
    for topic in df['topic'].unique():    
        knn_df = df.loc[df['topic']==topic].reset_index(drop=True)

        map_idx_to_stance = dict(zip(knn_df.index, knn_df.stance))
        map_idx_to_tweet = dict(zip(knn_df.index, knn_df.tweet_id))

        X = knn_df[[c for c in knn_df.columns if 'emb' in c]].values
        knn = NearestNeighbors(n_neighbors=n+1, metric='euclidean', algorithm='brute').fit(X) # n+1 since the nearest neighbor of each point is itself
        distances, indices = knn.kneighbors(X)
        distances = distances[:, 1:].tolist() # remove distance to itself

        # Convert to dataframe
        indices_df = pd.DataFrame(indices, 
                                  columns=["actual_class" if i == 0 else f"neighbor_{i}" for i in range(indices.shape[-1])])

        neighbors = indices_df.applymap(map_idx_to_tweet.get) # extract nearest neighbors tweet IDs
        classes = indices_df.applymap(map_idx_to_stance.get) # map tweet IDs to their labels
        classes['majority_class'] = classes[[i for i in classes.columns if "neighbor" in i]].mode(axis=1) # majority vote
        classes['topic'] = topic
        classes['tweet_id'] = knn_df.tweet_id
        results.append(classes)
    final_df = pd.concat(results, ignore_index=True)
    return final_df


def compute_nn_metrics(paths, test_df=None, n=1, metric="viewpoint", pca_norm=False, n_components=30):
    '''
    Returns a dataframe with NN metric
    Inputs:
        - paths: path to embeddings for evaluation
        - test_df: annotated data
        - n: number of neighbors
        - metric: viewpoint or topic
        - pca_norm: if True then apply PCA
        - n_components: only if PCA norm is True
    '''
    results = [] 
    for emb_path, model_id in tqdm(paths):

        emb_df = pd.read_csv(emb_path, dtype={'tweet_id':str}) # load embeddings
        if pca_norm: # Only apply if PCA is needed
            emb_df = apply_pca(emb_df, n=n_components)
        df = pd.merge(emb_df, test_df[['tweet_id', 'topic', 'stance']], on='tweet_id', how='left')
        if metric == 'viewpoint':
            final_df = compute_knn_viewpoint(df=df, n=n)
        elif metric == 'topic':
            final_df = compute_knn_topic(df=df, n=n)
        else:
            print('Metric is not implemented...')
            
        final_df['model_id'] = model_id
        final_df["N"] = n
        results.append(final_df)
        
    df = pd.concat(results, ignore_index=True)
    return df


def bertopic_load_embeddings(path, test_df=None):
    '''
    Return ids, labels, docs, and embeddings
    '''
    
    emb_df = pd.read_csv(path, dtype={'tweet_id':str})
    df = pd.merge(emb_df, test_df[['tweet_id', 'full_text', 'label']], on='tweet_id', how='left')
    
    X = df[[c for c in df.columns if 'emb' in c]].values
    
    return df['tweet_id'].values, df['label'].values, df['full_text'].values, X
