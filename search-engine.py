# Daniel, Ramirez G.
# dgr2815
# 2019-09-26
#---------#---------#---------#---------#---------#--------#
import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#---------#---------#---------#---------#---------#--------#
corpusroot = './presidential_debates'
# token pattern definition, meant to identify words containing only
# ASCII a to z upper and lowercase. Pattern does not include empty string.
tokenizer = RegexpTokenizer( r'[a-zA-Z]+' )
# retrieve list of English stopwords
english_stopwords = stopwords.words( 'english' )
# create stemmer
stemmer = PorterStemmer()
#---------#---------#---------#---------#---------#--------#

def get_documents( corpusroot ) :
    '''Constructs list of strings to represent documents and list of strings
        to represent filenames.'''
    # directory containing unprocessed corpus
    docs = []
    files = []
    # iterate over files in 'corpusroot'
    for filename in os.listdir( corpusroot ) :
        # store filenames in list
        files.append( filename )
        # store lowercase-converted contents of files in list
        with open( os.path.join( corpusroot, filename ), "r", encoding='UTF-8' ) as f :
            doc = f.read()
        docs.append( doc.lower() )
        
    return docs, files
        
def tokenize_documents( docs ) :
    '''Tokenize documents using simple regular expression for a to z characters
        upper and lower case'''
    tokenized_docs = []
    # tokenize contents of all documents according to pattern
    # store tokenized content in list
    for doc in docs :
        tokenized_doc = tokenizer.tokenize( doc )
        tokenized_docs.append( tokenized_doc )
    
    return tokenized_docs

def stem_documents_and_create_vectors( docs ) :
    '''Stem documents using porter stemming while filtering out english 
        stopwords and creating empty tf_idf_vectors for later use.'''
    tf_idf_vectors = []
    stemmed_docs = []
    # iterate over documents
    for doc in documents :
        tf_idf_vector = {}
        stemmed_doc = []
        # iterate over terms to create list of stemmed non-stopword terms
        # create "vector" of terms and their tf_idf weights, represented as dict
        #   tf_idf weights not calculated here
        for token in doc :
            if not ( token in english_stopwords ) :
                stemmed_token = stemmer.stem( token )
                tf_idf_vector[ stemmed_token ] = 0
                stemmed_doc.append( stemmed_token )
        tf_idf_vectors.append( tf_idf_vector )
        stemmed_docs.append( stemmed_doc )
    
    return stemmed_docs, tf_idf_vectors
        

def normalize_tfidf_vector( vector ) :
    l2_norm = 0
    # Calculate the L2 norm for the vector
    for value in vector.values() :
        l2_norm = l2_norm + value ** 2
    l2_norm = math.sqrt( l2_norm )
    
    # Normalize vector using the L2 norm
    for key, value in vector.items() :
        vector[ key ] = value / l2_norm
    
    return vector


def weigh_documents( docs, vectors ) :
    num_of_docs = len( docs )
    print( 'completed [ out of ' + str( num_of_docs ) + ' ]: ', end='' )
    i = 0
    # iterate over documents
    for doc in docs :
        # iterate over terms to calculate tf-idf weight for each
        for term in doc :
            # calculate tf-idf weight
            weight = gettf( i, term ) * getidf( term )
            # add to tf_idf vector for the specific doc
            vectors[ i ][ term ] = weight
            # calulating vector magnitude sqrt( weight_i ^ 2 ) for all i in vector
        vectors[ i ] = normalize_tfidf_vector( vectors[ i ] )       
        if ( i+1 ) < num_of_docs :
            print( str( i+1 ) + ', ', end='' )
        else :
            print( str( i+1 ) )
        i += 1    
        
    return vectors

def build_postings_lists( docs, _filenames, vectors ) :

    terms = []
    # get list of all terms in corpus by iterating over all documents
    for doc in docs :
        for term in doc :
            terms.append( term )
    
    postings = []
    # iterate over all terms to create posting list for each
    for term in terms :
        posting = []
        # iterate over tf_idf_vectors to get values for postings list
        for vector in vectors :
            filename = _filenames[ vectors.index( vector ) ]
            try :
                value = vector[ term ]
            except KeyError :
                value = 0
            posting.append( ( filename, value ) )
        posting.sort( key=lambda tup: tup[1], reverse=True )
        postings.append( ( term, posting ) )
    return postings

def naive_query( qstring ) :
    '''naive approach to query retrieval. performs cosine similarity test
        against every single document instead of implementing postings list'''
    filename = 'None'
    score = 0
    
    # get all terms in the corpus for later use
    terms = []
    for document in documents :
        for term in document :
            terms.append( term )
    
    # prepare qstring to be analyzed, by tokenizing and stemming
    tokens = tokenizer.tokenize( qstring )
    filtered_tokens = []
    for token in tokens :
        # filter out stopwords and words not in the corpus
        if not ( token in english_stopwords ) :
            if token in terms :
                filtered_tokens.append( stemmer.stem( token ) )

    # if all tokens have been ignored as the result of not being in the corpus
    # return early, indicating that there is nothing to be found
    if len( filtered_tokens ) == 0 :
        return ( filename, score )

    # create vector of query tf scores
    query_vector = querytf( filtered_tokens )
    
    doc_scores = []
    for i in range( len( filenames ) ) :
        doc_vector = []
        for token in filtered_tokens :
            try :
                weight = document_weights[i][ token ]
            except KeyError :
                weight = 0.0
            doc_vector.append( weight )
        doc_score = cosine_similarity( query_vector, doc_vector )
        doc_scores.append( doc_score )
    
    max_score = 0
    for i in range( len( doc_scores ) ) :
        if doc_scores[i] > max_score :
            max_score = doc_scores[i]
            filename = filenames[i]
        
    return ( filename, max_score )

def query( qstring ) : 
    '''return a tuple in the form of (filename of the document, score), 
        where the document is the query answer with respect to "qstring" 
        according to (7.5). If no document contains any token in the query, 
        return ("None",0). If we need more than 10 elements from each 
        posting list, return ("fetch more",0).'''
    filename = 'None'
    score = 0
    
    # get all terms in the corpus for later use
    terms = []
    for document in documents :
        for term in document :
            terms.append( term )
    
    # prepare qstring to be analyzed, by tokenizing and stemming
    tokens = tokenizer.tokenize( qstring )
    filtered_tokens = []
    for token in tokens :
        # filter out stopwords and words not in the corpus
        if not ( token in english_stopwords ) :
            if token in terms :
                filtered_tokens.append( stemmer.stem( token ) )

    # if all tokens have been ignored as the result of not being in the corpus
    # return early, indicating that there is nothing to be found
    if len( filtered_tokens ) == 0 :
        return ( filename, score )

    # create vector of query tf scores
    query_vector = querytf( filtered_tokens )
    
    query_postings = []
    # get list of postings for each token that appears in the corpus
    for token in filtered_tokens :
        query_postings.append( get_top10_postings( token ) )
    
    intersection = query_postings[0].keys()
    for postings in query_postings :
        intersection = intersection & postings.keys()
    
    filename = list( intersection )[0]
    
    doc_vector = []
    for i in range( len( filtered_tokens ) ) :
        value = query_postings[ i ][ filename ]
        doc_vector.append( value )
    
    score = cosine_similarity( query_vector, doc_vector )
        
    return ( filename, score )
    

def cosine_similarity( vector1, vector2 ) :
    '''expects two lists of floating points or integers of the same length
        returns cosine similarity of them'''
    result = 0
    for i in range( len( vector1 ) ) :
        result = result + ( vector1[i] * vector2[i] )
    return result

def querytf( qtokens ) :
    '''gets tokenized and stemmed query string, returns normalized vector of
        term frequency values'''
    # create vector of raw term frequency of terms with respect to query
    vector = []
    for token in qtokens :
        term_freq = qtokens.count( token )
        vector.append( term_freq )
    
    # normalize with L2 norm
    l2_norm = 0
    # Calculate the L2 norm for the vector
    for value in vector :
        l2_norm = l2_norm + value ** 2
    l2_norm = math.sqrt( l2_norm )
    
    # Normalize vector using the L2 norm
    for i in range( len( vector ) ) :
        vector[i] = vector[i] / l2_norm
        
    return vector
    

def gettf( index, token ) :
    '''get term frequency of a token in a given document, represented by
        the index in the list of documents. this is a helper function, not
        meant to be used on its own'''
    term_freq = documents[ index ].count( token )
    
    if not term_freq == 0 :
        return ( 1 + math.log10( term_freq ) )
    else :
        return 0

def getidf( token ) : 
    '''return the inverse document frequency of a token. If the token 
        doesn't exist in the corpus, return -1. The parameter 'token' is 
        already stemmed. Note the differences between getidf("hispan") and 
        getidf("hispanic") in the examples below.'''
    N = len( documents )
    doc_freq = 0
    # get list of all terms in corpus by iterating over all documents
    for vector in document_weights :
        if token in vector :
            doc_freq += 1
    
    if not doc_freq == 0 :
        return math.log10( N / doc_freq )
    else :
        return -1

def getweight( filename, token ) : 
    '''return the TF-IDF weight of a token in the document named 
        'filename'. If the token doesn't exist in the document, return 0. 
        The parameter 'token' is already stemmed. Note that both 
        getweight("1960-10-21.txt","reason") and 
        getweight("2012-10-16.txt","hispanic") return 0, but for different 
        reasons. Return -1 for filename error.'''
    try :
        index = filenames.index( filename )
    except ValueError :
        return -1
    
    if token in document_weights[ index ] :
        return document_weights[ index ][ token ]
    else :
        return 0

def get_top10_postings( token ) :
    '''given a token return the top 10 postings lists including any that
        may have a score of zero. return empty dictionary 
    '''
    top10_postings = {}
    postings = []
    
    # find corresponding postings list for token given
    for i in range( len( postings_lists ) ) :
        if token == postings_lists[i][0] :
            postings = postings_lists[i][1]
    
    if len( postings ) == 0 :
        return {}
    
    # get top 10 postings from sorted postings list, return as dictionary
    for i in range( 10 ) :
        top10_postings[ postings[i][0] ] = postings[i][1]
    return top10_postings

#---------#---------#---------#---------#---------#--------#
print( 'getting documents from ' + corpusroot + '...' )
documents, filenames = get_documents( corpusroot )
print( 'tokenizing documents...' )
documents = tokenize_documents( documents )
print( 'stemming documents...' )
documents, document_weights = stem_documents_and_create_vectors( documents )
print( 'calculating tf-idf vectors...' )
document_weights = weigh_documents( documents, document_weights )
print( 'building postings lists...' )
postings_lists = build_postings_lists( documents, filenames, document_weights )


if __name__ == '__main__' :
    while 1 :
        
        try :
            choice = int( input( '''query - 0
unfinished postings list query - 1
getidf - 2
getweight - 3
gettop10postings - 4
EXIT - -1
enter number> ''' ) )
        except TypeError :
            continue
        except ValueError :
            continue
        
        if choice == 0 :
            qstring = input( 'enter query> ' ).lower()
            print( 'getting results of query ' + qstring )
            print( '(%s, %.12f)' % naive_query( qstring ) )
        elif choice == 1 :
            qstring = input( 'enter query> ' ).lower()
            print( 'getting results of query ' + qstring )
            print( '(%s, %.12f)' % query( qstring ) )
        elif choice == 2 :
            token = input( 'enter token> ' ).lower()
            print( 'getting idf weight of ' + token )
            print( '%.12f' % getidf( token ))
        elif choice == 3 :
            token = input( 'enter token> ' ).lower()
            filename = input( 'enter filename> ' )
            print( 'getting tf-idf weight of ' + token + ' from ' + filename )
            print( '%.12f' % getweight( filename, token ))
        elif choice == 4 :
            token = input( 'enter token> ' ).lower()
            print( 'getting postings list of ' + token )
            print( get_top10_postings( token ) )
#        elif choice == 5 :
#            print("(%s, %.12f)" % query("health insurance wall street"))
#            print( ' (2012-10-03.txt, 0.033877975254)')
#            print("(%s, %.12f)" % query("particular constitutional amendment"))
#            print( ' (fetch more, 0.000000000000)')
#            print("(%s, %.12f)" % query("terror attack"))
#            print( ' (2004-09-30.txt, 0.026893338131)')
#            print("(%s, %.12f)" % query("vector entropy"))
#            print( ' (None, 0.000000000000)')
#            
#            print("%.12f" % getweight("2012-10-03.txt","health"))
#            print( ' 0.008528366190')
#            print("%.12f" % getweight("1960-10-21.txt","reason"))
#            print( ' 0.000000000000')
#            print("%.12f" % getweight("1976-10-22.txt","agenda"))
#            print( ' 0.012683891289')
#            print("%.12f" % getweight("2012-10-16.txt","hispan"))
#            print( ' 0.023489163449')
#            print("%.12f" % getweight("2012-10-16.txt","hispanic"))
#            print( ' 0.000000000000')
#
#                
#            print("%.12f" % getidf("health"))
#            print( ' 0.079181246048')
#            print("%.12f" % getidf("agenda"))
#            print( ' 0.363177902413')
#            print("%.12f" % getidf("vector"))
#            print( ' -1.000000000000' )
#            print("%.12f" % getidf("reason"))
#            print( ' 0.000000000000')
#            print("%.12f" % getidf("hispan"))
#            print( ' 0.632023214705' )
#            print("%.12f" % getidf("hispanic"))
#            print( ' -1.000000000000' )
        else :
            break
        
#---------#---------#---------#---------#---------#--------#
