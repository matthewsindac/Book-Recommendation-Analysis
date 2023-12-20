import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns

bookratings = pd.read_csv("C:/Users/Matthew/IdeaProjects/CS 24200 Project 3/venv/BX-Book-Ratings.csv",
                          sep = ';', encoding = 'latin-1')
books = pd.read_csv("C:/Users/Matthew/IdeaProjects/CS 24200 Project 3/venv/BX-Books.csv",
                    sep = ';', encoding = 'latin-1', quotechar = '"', on_bad_lines = 'skip')
users = pd.read_csv("C:/Users/Matthew/IdeaProjects/CS 24200 Project 3/venv/BX-Users.csv",
                    sep = ';', encoding = 'latin-1', quotechar = '"')

isbnbooks = bookratings.groupby('ISBN')['Book-Rating'].mean().reset_index()
merged = pd.merge(bookratings, isbnbooks, on = 'ISBN', how = 'left', suffixes = ('', '_mean'))
bookratings.loc[bookratings['Book-Rating'] == 0, 'Book-Rating'] = merged['Book-Rating_mean'].astype(int)

isbncounts = bookratings['ISBN'].value_counts().reset_index()
isbncounts.columns = ['ISBN', 'ISBN_RatingCount']

usercounts = bookratings['User-ID'].value_counts().reset_index()
usercounts.columns = ['User-ID', 'User_RatesCount']

filteredisbncounts = isbncounts.loc[isbncounts['ISBN_RatingCount'] >= 200]\
    .sort_values(by = 'ISBN_RatingCount', ascending = False)
filteredusercounts = usercounts.loc[usercounts['User_RatesCount'] >= 5]\
    .sort_values(by = 'User_RatesCount', ascending = False)

newdata = bookratings.merge(filteredisbncounts, on = 'ISBN')\
    .merge(filteredusercounts, on = 'User-ID')\
    .sort_values(by = ['ISBN_RatingCount', 'User_RatesCount'], ascending = [False, False])\
    .reset_index(drop = True)

newdata.to_csv('Data Of UserIDS, Book Ratings, Amount of Book Ratings Per Book, Amount of User Ratings Per User')

usercounts = newdata[['ISBN', 'ISBN_RatingCount']].drop_duplicates()
sortedusercounts = usercounts.sort_values(by='ISBN_RatingCount', ascending=False)
topthreeisbns = sortedusercounts.head(3)

topthreeisbns.to_csv('Top 3 Books Ratings')

match = pd.merge(topthreeisbns['ISBN'], books, on = 'ISBN', how = 'inner')

match.to_csv('Top 3 Books Titles')

users['Country'] = users['Location'].str.split(',').str[-1].str.strip()

newdata2 = bookratings.merge(filteredisbncounts, on = 'ISBN').merge(filteredusercounts, on = 'User-ID') \
    .merge(users[['User-ID', 'Country']], on = 'User-ID')\
    .sort_values(by = ['ISBN_RatingCount', 'User_RatesCount']).reset_index(drop = True)

userratingsmatrix = newdata2.pivot_table\
    (index = 'User-ID', columns = 'ISBN', values = 'Book-Rating')

def klistclustering(newdata):
    newdata = newdata.merge(users[['User-ID', 'Age']], on = 'User-ID', how = 'left')
    newdata = newdata.dropna(subset = ['Age'])
    ndata = newdata.drop(['User-ID', 'ISBN'], axis = 1)

    klist = [2,4,8,16,32,64,128]
    inertiavalues = []

    for k in klist:
        kmeans = KMeans(n_clusters = k, n_init = 10)
        kmeans.fit(ndata)
        inertiavalues.append(kmeans.inertia_)

    print(inertiavalues)

    plt.scatter(klist, inertiavalues, marker = 'o')
    plt.xlabel('K Cluster Number')
    plt.ylabel('Inertia')
    plt.title('Inertia Values for Each K')
    plt.show()

klistclustering(newdata)

def chosenvaluek(newdata):
    newdata = newdata.merge(users[['User-ID', 'Age']], on = 'User-ID', how = 'left')
    newdata = newdata.dropna(subset = ['Age'])
    ndata = newdata.drop(['User-ID', 'ISBN'], axis = 1)
    kmeans = KMeans(n_clusters = 16, n_init = 10)
    newdata['Cluster'] = kmeans.fit_predict(ndata)

    clusterratings = newdata.groupby\
        (['Cluster', 'ISBN'])['Book-Rating'].mean().reset_index()

    clusterratings2 = newdata.groupby \
        (['User-ID', 'Cluster', 'ISBN'])['Book-Rating'].mean().reset_index()

    clusterratings['Rank'] = clusterratings.groupby('Cluster')['Book-Rating']\
        .rank(ascending = False, method = 'max')

    clusterratings.to_csv('A - CLUSTER RATINGS')

    topthreebooks = clusterratings[clusterratings['Rank'] <= 3]
    topthreebooks = topthreebooks.drop(['Rank'], axis = 1)

    topthreebooks.to_csv('A - TOP 3 EACH CLUSTER')

    top3booksdetails = pd.merge(topthreebooks, books, on = 'ISBN', how = 'left')
    top3booksdetails = top3booksdetails.drop(['Book-Author', 'Year-Of-Publication', 'Publisher',
                           'Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis = 1)

    top3booksdetails.to_csv('A - TOP 3 TITLES EACH CLUSTER')

    youngestandoldest = newdata.groupby('Cluster')['Age'].agg(['min', 'max']).reset_index()
    meanage = newdata.groupby('Cluster')['Age'].mean().reset_index()
    ages = pd.merge(meanage, youngestandoldest, on = 'Cluster', how = 'left')

    ages.rename(columns = {'Age': 'Mean Age', 'min': 'Minimum Age', 'max': 'Maximum Age'}, inplace = True)

    ages.to_csv('A - LOWEST AGE AND TOP AGE PER CLUSTER')

    return clusterratings2

chosenvaluek(newdata)


def pca(userratingsmatrix, users):
    transposedmatrix = userratingsmatrix.transpose().fillna(0)
    X = transposedmatrix.values
    X_scaled = preprocessing.scale(X, with_std = False)

    pca = PCA(n_components = 2)
    X_trans = pca.fit_transform(X_scaled)

    pcadata = pd.DataFrame(data = X_trans, columns = ['PCA_Component_1', 'PCA_Component_2']
                           , index = transposedmatrix.index)
    pcadata.reset_index(inplace = True)

    pcadata.to_csv('ISBN and PCA Results')

    unique = pcadata['ISBN'].unique()
    thecolors = sns.color_palette('husl', n_colors = len(unique))
    mapcolors = dict(zip(unique, thecolors))
    pcadata['Color'] = pcadata['ISBN'].map(mapcolors)

    plt.scatter(X_trans[:, 0], X_trans[:, 1], c = pcadata['Color'])
    plt.title('PCA Results (k = 2)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c = pcadata['Color'])
    plt.title('Original Data')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.show()

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    eightypercent = np.argmax(cumulative >= 0.8) + 1
    fortypercent = np.argmax(cumulative >= 0.4) + 1

    # print(explained)
    # print(cumulative)
    print(eightypercent)
    print(fortypercent)

pca(userratingsmatrix, users)


def svd(userratingsmatrix):
    userratingsmatrix = userratingsmatrix.fillna(0)

    model = PCA(n_components = 128)
    model.fit(userratingsmatrix)
    singularvalues = model.singular_values_

    plt.plot(singularvalues, marker='o')
    plt.xlabel('Component Number')
    plt.ylabel('Singular Value')
    plt.title('Singular Values for k = 128')
    plt.show()

    klist = [2, 4, 8, 16, 32, 64, 128]
    explainedvariances = []

    for k in klist:
        newmodel = PCA(n_components = k)
        newmodel.fit(userratingsmatrix)
        explainedvariances.append(np.sum(newmodel.explained_variance_ratio_))

    print(explainedvariances)

    modelk2 = PCA(n_components = 2)
    transformedmodelk2 = modelk2.fit_transform(userratingsmatrix)

    svd = pd.DataFrame(data = transformedmodelk2, columns = ['Component_1', 'Component_2'],
                       index = userratingsmatrix.index)
    svd.reset_index(inplace = True)

    clusters = chosenvaluek(newdata)

    mergeddata = pd.merge(svd, clusters[['User-ID', 'Cluster']],
                          on = 'User-ID', how = 'left')
    mergeddata = mergeddata.dropna(subset = ['Cluster'])

    plt.scatter(mergeddata['Component_1'], mergeddata['Component_2'],
                c = mergeddata['Cluster'])
    plt.title('Scatter Plot of Component_1 vs Component_2 (k = 2)')
    plt.xlabel('Component_1')
    plt.ylabel('Component_2')
    plt.show()

svd(userratingsmatrix)
