######
# This script uses LDA and TFIDF to categorize
# a document collection, then uses the same procedure
# to find the best results for any given search query
# the results are saved in a *.trac_eval file
import argparse
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from xml.dom import minidom

#reads the document colleciton and search queries
def read_data():
    data_path = "data/ie1_collection.trec"
    search_path = "data/ie1_queries.trec"
    try:
        data_doc = minidom.parse(data_path)
        query_doc = minidom.parse(search_path)
        return 1, data_doc, query_doc
    except:
        print("The required *.trec files were not found. Make sure they are located at the following places")
        print(data_path)
        print(search_path)
        return 0, {}, {}

###Data preprocessing
#extracts text from XML element
def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


def parse_data(data_doc, query_doc):

    data_dictionary = {}
    query_dictionary = {}

    doclist = data_doc.getElementsByTagName('DOC')
    querylist = query_doc.getElementsByTagName('DOC')
    if args.debug:
        print('Parsing all documents...')
    for doc in doclist:
        recordId = getText(doc.getElementsByTagName("recordId")[0].childNodes)
        text = getText(doc.getElementsByTagName("text")[0].childNodes)
        data_dictionary[recordId] = text

    if args.debug:
        print('Parsing all search queries...')
    for query in querylist:
        recordId = getText(query.getElementsByTagName("recordId")[0].childNodes)
        text = getText(query.getElementsByTagName("text")[0].childNodes)
        query_dictionary[recordId] = text
    
    if args.debug:
        print('length data_dictionary ' + str(len(data_dictionary)))
        print('length query_dictionary ' + str(len(query_dictionary)))
    
    return data_dictionary, query_dictionary

def sort_data_to_generated_topics(data_dictionary, vectorizer, predicted_topics):
    #Now we have to sort our data into the trained topics
    data_topic_dictionary = {}
    if args.debug:
        print('Generating topic structure...')
    for index in predicted_topics:
        if data_topic_dictionary.get(str(index)) is None:
            data_topic_dictionary[str(index)] = []

    if args.debug:
        print('Sorting data into topics...')
    for index, value in enumerate(predicted_topics):
        data_topic_dictionary[str(value)].append({"key": list(data_dictionary.keys())[index],
        "value": vectorizer.transform([list(data_dictionary.values())[index]])})
    
    return data_topic_dictionary


###Search query application and scoring
def score_search_queries_and_save(query_dictionary, data_topic_dictionary, vectorizer, lda, filename, groupname='group3'):
    max_ranks = 51

    with open(filename + '.trec_eval', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',)
        if args.debug:
            print('FileWriter is open and ready to write')

        #write header of the document
        writer.writerow(['QueryID', 'Iteration', 'Dok.Nummer', 'Rang', 'Score', 'System'])
        
        if args.debug:
            print('Looping through each query')
        for query_id, search_term in query_dictionary.items():
            tfst = vectorizer.transform([search_term])
            topic = lda.transform(tfst)
            categorized_topic = topic.argmax()
            possible_documents = data_topic_dictionary[str(categorized_topic)]
            documents_with_distance = {}
            for document in possible_documents:
                distance = pairwise_distances(tfst, document['value']).tolist()[0][0]
                documents_with_distance[document['key']] = distance
            ordered_doc = sorted(documents_with_distance.items(), key=lambda x: x[1])

            for index, od in enumerate(ordered_doc):
                writer.writerow([query_id,'Q0', od[0], index+1, od[1], groupname])
                if index == max_ranks-1:
                    break
        #slice_length = len(ordered_doc) if len(ordered_doc < 50) else 50
        #ordered_doc = dict(itertools.islice(ordered_doc.items(), slice_length))
    if args.debug:
        print('Finished writing the .trac_eval file')


###Unsupervised learning
def create_vectorizer_with_data(data_dictionary):
    stopwords_language = 'english' if args.stopwords else None
    max_f = args.maxfeatures
    if args.debug:
        print('Stopwords language selected is: ' + str(stopwords_language))
        print('Max features is set to: ' + str(max_f))

    if args.tfidf:
        if args.debug:
            print('Using TfIdf Vectorizer!')
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=4, stop_words=stopwords_language, max_features=max_f, ngram_range=(1,3))
        vectorized_data = vectorizer.fit_transform(data_dictionary.values())
    else:
        if args.debug:
            print('Using Count Vectorizer!')
        vectorizer = CountVectorizer(max_df=0.95, min_df=4, stop_words=stopwords_language, max_features=max_f, ngram_range=(1,3))
        vectorized_data = vectorizer.fit_transform(data_dictionary.values())
    
    return vectorizer, vectorized_data

def create_lda_and_categorize(vectorized_data):
    n_topics = args.topics
    r_state = args.randomstate
    if args.debug:
        print('Amount of topics is: ' + str(n_topics))
        print('Random state is: ' + str(r_state))

    lda = LatentDirichletAllocation(n_components = n_topics, random_state = r_state)

    transformed_data = lda.fit_transform(vectorized_data)
    predicted_topics = np.argmax(transformed_data, axis=1)
    return lda, predicted_topics


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Create a document search engine using unsupervised learning")
    parser.add_argument('-t', '--topics', help="Set how many topics can be categorized", type=int, default=10)
    parser.add_argument('-mf', '--maxfeatures', help="Sets max features for the vectorizer", type=int, default=5000)
    parser.add_argument('-r', '--randomstate', help="Sets a given random state for LDA", type=int, default=69069)
    parser.add_argument('-f', '--file', help="Name the trec_eval  file", default="ranking")
    parser.add_argument('-d', '--debug', help="Turn on debug messages", action="store_true")
    parser.add_argument('-idf', '--tfidf', help="Select TFIDF vectorizer instead of count vectorizer", action="store_true")
    parser.add_argument('-s', '--stopwords', help="Sets english for stopwords, default is none", action="store_true")

    return parser.parse_args(argv)

def main(argv):

    print('Document Search Engine with unsupervised ML methods')
    print('---------------------------------------------------')
    global args 
    args = parse_args(argv)
    print('Parsing dataset')
    state, data_doc, query_doc = read_data()
    if not state:
        print("Terminating application.")
        exit()

    data_dictionary, query_dictionary = parse_data(data_doc, query_doc)
    print('Data parsed')
    print('---------------------------------------------------')
    print('Beginning training')
    vectorizer, vectorized_data = create_vectorizer_with_data(data_dictionary)
    lda, predicted_topics = create_lda_and_categorize(vectorized_data)
    print('Finished training')
    print('---------------------------------------------------')
    print('Sorting data in trained structure')
    data_topic_dictionary = sort_data_to_generated_topics(data_dictionary, vectorizer, predicted_topics)
    print('Finished sorting')
    print('---------------------------------------------------')
    print('Executing all search queries and saving as ' + args.file + '.trac_eval')
    score_search_queries_and_save(query_dictionary, data_topic_dictionary, vectorizer, lda, args.file)
    print('Finished query execution.')
    print('---------------------------------------------------')
        
if __name__ == '__main__':
    import sys
    exit(main(sys.argv[1:]))