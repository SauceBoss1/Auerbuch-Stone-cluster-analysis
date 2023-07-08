from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, FunctionTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import argparse

class ClusterTMM:
    kmeans_kwargs = {
        "init" : "k-means++",
        "random_state" : 1,
        "n_init": 'auto',
        "max_iter": 400
    }

    def __init__(self, file, names=[], kmeans_kwargs = kmeans_kwargs, scaler = StandardScaler()):
        self.file = file
        self.names = names
        self.kmeans_kwargs = kmeans_kwargs
        self.desc, self.dataset = self.getDataset()
        self.scaler = scaler

        self.scaled_data = self.scaler.fit_transform(self.dataset)
        self.dataset = pd.DataFrame(self.scaled_data, columns=names)
        self.clusteredDF = None

        
    def getDataset(self):
        df = pd.read_csv(self.file, sep='\t')
        return df.iloc[:,:5], df.iloc[:,5:]
        
    def plotElbow(self, outName = None):
        distortions = []

        for k in range(1,11):
            kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
            kmeans.fit_transform(self.scaled_data)
            distortions.append(kmeans.inertia_)
        
        plt.plot(range(1,11), distortions)
        plt.xticks(range(1,11))
        plt.xlabel('# of clusters')
        plt.ylabel('distortions')

        if outName is not None:
            plt.savefig(outName, dpi=720, format='pdf')
        plt.show()

    def cluster(self, n_clusters, showCenters = False, showDist = False, sort = True):
        kmeans = KMeans(n_clusters=n_clusters, **self.kmeans_kwargs)
        kmeans.fit_transform(self.scaled_data)
        y = kmeans.predict(self.scaled_data)

        df = self.desc
        df = pd.concat([df, self.dataset], axis=1)
        df['cluster'] = y

        if showDist:
            x = plt.scatter(df.iloc[:, 5],df.iloc[:, 8], c=df['cluster'], cmap=plt.cm.tab20b)
            x = plt.scatter(df.iloc[:, 6],df.iloc[:, 8], c=df['cluster'], cmap=plt.cm.tab20b)
            x = plt.scatter(df.iloc[:, 7],df.iloc[:, 8], c=df['cluster'], cmap=plt.cm.tab20b)
            if showCenters:
                centers = kmeans.cluster_centers_
                for i in range(n_clusters):
                    plt.scatter(centers[:, 2], centers[:, i], c='black')

            plt.show()
        if sort:
            df = df.sort_values('cluster')

        df.drop(df.columns[0], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.clusteredDF = df
        return df
    
    def dropCols(self, columns = []):
        self.dataset.drop(columns, inplace=True, axis=1)

    def createHeatmap(self, title: str, export = False, outName = '', file = None):
        if self.clusteredDF is not None and file is None:
            df = self.clusteredDF
            rowInd = df.index
            df = df.iloc[:,4:]
        else:
            df = pd.read_csv(file, sep='\t')
            rowInd = df.iloc[:,0]
            df = df.iloc[:,5:]
        
        n_clusters = len((df['cluster']).unique())
        n_cols = len(df.columns) - 1

        clusterLines = [0]
        for i in range(n_clusters):
            inDf = df[df['cluster']==i]
            clusterLines.append(inDf.index[-1])
        
        midpts = []
        for i in range(1,n_clusters+1):
            midpt = (clusterLines[i] + clusterLines[i-1]) // 2
            midpts.append(midpt)
        
        clusterLines = clusterLines[1:]

        labels = [f'cluster {i}' for i in range(1, n_clusters+1)]
        
        #df = df.iloc[:, :-1]

        plt.figure(figsize=(1080,1080))
        fig, axs = plt.subplots(1,n_cols)
        

        for i, axis in enumerate(axs):
            newdf = pd.DataFrame(df.iloc[:,i], index=rowInd)
            if axis != axs[-1]:
                h = sns.heatmap(newdf, cbar=False, cmap='plasma', ax=axis)
            else:
                h = sns.heatmap(newdf, ax=axis, cbar=True, cbar_kws={'label': 'Normalized Mean TMM value'}, cmap='plasma')

            if axis == axs[0]:
                h.set_yticks(midpts)
                h.set_yticklabels(labels)
            else:
                h.set_yticks([])
            h.set_ylabel('')

        for ax in axs:
            ax.hlines(clusterLines, *ax.get_xlim(), color='white')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        #fig.suptitle(title, fontsize=16)

        if export and outName != '':
            fig.savefig(outName, format='pdf', dpi=1080)
        plt.show()

def main():
    file = 'master means/26C_mean_masterFile.tsv'
    names = ['pil', 'pcnB', 'pil-pcnB', 'WT']
    dropColumnss = ['pil','pil-pcnB']

    ctmm = ClusterTMM(file=file, names=names, scaler=Normalizer())
    ctmm.dropCols(columns=dropColumnss)

    #ctmm.plotElbow(outName='test.pdf')
    ctmm.cluster(3).to_csv('test.tsv', sep='\t')
    ctmm.createHeatmap(file='test.tsv', title='test.pdf', export=True, outName='test.pdf')

if __name__ == '__main__':
    main()
