# Topic Modeling Using LSA and LDA

<br>
The t-SNE clusters of the 10 topics recognized by the LSA technique. We can see that this technique does not categorize the passages into topics very well.
<p align="center">
<img width="715" alt="Screenshot 2022-02-06 at 4 35 09 PM" src="https://user-images.githubusercontent.com/46462603/152706423-373ce6a3-365f-461b-aca0-3e34b47b0c10.png">
</p>

<br>
The t-SNE clusters of the 10 topics recognized by the LDA technique. The passages are categorized into 10 topics more evenly.
<p align="center">
<img width="706" alt="Screenshot 2022-02-06 at 4 35 24 PM" src="https://user-images.githubusercontent.com/46462603/152706553-38ce9ced-408f-44d2-b16a-d09034c48e2a.png">
</p>

<br>
The interactive display showing the 10 topics and the most frequently occurred words. 
<p align="center">
<img width="1221" alt="Screenshot 2022-02-06 at 4 35 44 PM" src="https://user-images.githubusercontent.com/46462603/152706655-1122f117-a43c-4f56-be52-30a388804fcc.png">
</p>

<br>
If you hover over a topic (a circle with a number in it on the left side of the display), the bar charts on the right will be updated, showing the most frequently occurred words in the chosen topic in red.
<p align="center">
<img width="1223" alt="Screenshot 2022-02-06 at 4 43 20 PM" src="https://user-images.githubusercontent.com/46462603/152706662-20f137e8-936f-4028-9214-d81d8f76d9f6.png">
</p>

## Goal
- To implement topic modeling to categorize the passages in the RACE dataset into 10 topics and determine the most frequently used words in each topic.
- To compare the performances of LSA and LDA in topic modeling. 

## Dataset
The RACE dataset is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students. The dataset can be downloaded from <a href="https://www.cs.cmu.edu/~glai1/data/race/" target="_blank">here</a>. The dataset is too large to be uploaded to GitHub. 

## Code, files, or folders needed to run the program
- <a href="https://github.com/ZhengEnThan/Topic-Modeling/blob/main/Topic_Modeling_with_LSA_and_LDA.ipynb" target="_blank">Topic_Modeling_with_LSA_and_LDA.ipynb</a>
- documents.csv, which is the RACE dataset that can be downloaded from <a href="https://www.cs.cmu.edu/~glai1/data/race/" target="_blank">here</a>

## How to use the program
Simply run the program cell by cell to import the library, load the data, process and inspect the data, and then apply topic modeling using LSA as well as LDA on the data. 

There are comments in each code cell that tell you what the code does and the code cells are also grouped under different markdowns that specify what the group of code cells do.

If you want to use another dataset, edit the code cell directly below the markdown "Getting the data".

```python
documents = pd.read_csv("documents.csv")
print("There are ", len(documents), " documents in total.\n")
print("The first 5 rows of the dataset are:\n")
print(documents.head())
```

Change "documents.csv" in the code cell above to the file name of your dataset. Make sure that your dataset is such that there is only 1 column where each entry is a passage. If your dataset is not in the csv format, you would have to change the function pd.read_csv() as well. 

## What I have learned 
- Used nltk to remove stopwords, tokenize, and lemmatize the passages in the dataset.
- Used modules from sklearn.feature_extraction.text such as CountVectorizer and functions from numpy such as argsort() and flip() to inspect the properties of the dataset such as finding the top n words that occur the most in a given text.
- Used TextBlob from textblob to compute the frequency of each word type such as noun and adjective in a given text. 
- Used the TfidfVectorizer class from sklearn.feature_extraction.text to convert a collection of raw documents to a matrix of TF-IDF features and TruncatedSVD from sklearn.decomposition to do dimensionality reduction using truncated SVD. Computing a TF-IDF matrix and performing SVD on it are done in order to implement the LSA topic modeling technique.
- Used TSNE from sklearn.manifold to visualize high-dimensional data in 2D.
- Used WordCloud from wordcloud to display word clouds of a given text where the bigger a word is, the more frequently it appears in the text.
- Used LatentDirichletAllocation and RandomizedSearchCV to implement the LDA topic modeling technique and find the best parameters for the LDA model.
- Used bokeh and pyLDAvis to interactively display the plots such as the t-SNE cluster plots and plots that interpret the "distance" between each topic and the words that most frequently appear in each topic.

## Main libraries or modules used
- numpy
- pandas
- nltk
- textblob
- collections
- sklearn
- wordcloud
- bokeh
- pyLDAvis
- matplotlib
- seaborn

## Approaches
In order to reduce the size of the dataset, only the top 4000 most frequently occurred words are kept. To implement the LSA or Latent Semantic Analysis model, TF-IDF (term frequency-inverse document frequency) scores are computed where and stored in a matrix. The TF-IDF score of a word is high when it occurs frequently in a document but infrequently across the corpus or all documents. Therefore, compared to Count Vectorizer which simply keeps track of the occurrence of a word in a document, TF-IDF Vectorizer also keeps track of the 'exclusivity' of a given word. The resulting TF-IDF matrix will have documents as its rows and words as its columns. Each entry of the matrix will be the TF-IDF score that each word has for each document. 

Then, SVD or singular value decomposition is applied on the TF-IDF matrix computed. SVD will receive a matrix with documents as rows and words as columns, and produce 3 outputs. The 2 output matrices that we need are U and V. U will have documents as its rows and topics as its columns while V transposed will have topics as its rows and words as its columns. n_components is set to 10 when implementing TruncatedSVD so that the output matrices U and V will have 10 topics.

To better visualize the multidimensional relationships between the topics and documents as well as differences or "distances" between the topics, t-SNE is used. From the t-SNE plot of the LSA model result, we can see that LSA model does not categorize the documents into topics very well as there are about 90% of the documents that are categorized into the same topic. 

LDA or Latent Dirichlet Allocation is another topic modeling technique that is commonly used. It uses dirichlet priors for the document-topic and word-topic distributions, lending itself to better generalization. Assuming a generative model for a collection of documents, LDA tries to backtrack from the documents to find a set of topics that are likely to have generated the collection. Each document gets represented as a pattern of LDA topics. Making every document appear different enough to be separable and similar enough to be grouped. From its t-SNE plot, we can see that LDA can categorize the documents into topics more evenly. 

## Comments
Perplexity is commonly used to evaluate how well a probability model like LDA performs. Perplexity tells how well a given model can reproduce the statistics of the held-out data. However, it has been shown that optimizing perplexity may not yield human interpretable topics. Therefore, another method of evaluation is often used, which is topic coherence. Topic coherence measures the degree of semantic similarity between high scoring words in a given topic. The higher the coherence score, the better the model is at categorizing the documents into topics. 

## References
- Latent Semantic Analysis (LSA) <br>
https://blog.marketmuse.com/glossary/latent-semantic-analysis-definition/
- Data Processing: replace abbreviation of word <br>
https://www.kaggle.com/life2short/data-processing-replace-abbreviation-of-word/comments
- Topic Modeling with LSA, PLSA, LDA & lda2Vec <br>
https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
- Difference between PCA VS t-SNE <br>
https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/
- StatQuest: t-SNE, Clearly Explained <br>
https://www.youtube.com/watch?v=NEaUSP4YerM
- Lecture 47 — Singular Value Decomposition | Stanford University <br>
https://www.youtube.com/watch?v=P5mlg91as1c
- Latent Dirichlet Allocation: Intuition, math, implementation and visualisation with pyLDAvis <br>
https://towardsdatascience.com/latent-dirichlet-allocation-intuition-math-implementation-and-visualisation-63ccb616e094
- Evaluate Topic Models: Latent Dirichlet Allocation (LDA) <br>
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
- How does topic coherence score in LDA intuitively makes sense ? <br>
https://stats.stackexchange.com/questions/375062/how-does-topic-coherence-score-in-lda-intuitively-makes-sense
- Topic Modelling with LSA and LDA <br>
https://forestforthetree.com/statistics/2018/01/28/topic-modelling-with-lsa-and-lda.html
- Topic Modeling and Latent Dirichlet Allocation (LDA) in Python <br>
https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
- Introduction to Latent Dirichlet Allocation <br>
http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/
- LDA Topic Models <br>
https://www.youtube.com/watch?v=3mHy4OSyRf0

~ Project created in July 2021 ~
