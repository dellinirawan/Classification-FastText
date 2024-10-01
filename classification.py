import nltk
nltk.download('punkt_tab')
import streamlit as st
import pandas as pd
from owlready2 import *
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

# Classes (use your classes here like Pembobotan, Preprocessing, etc.)
class Pembobotan:
    def fasttext_with_dao_similarity(self, words, vocab):
        model = FastText(sentences=[words], vector_size=50, window=2, min_count=1, workers=4)
        similar_words_with_dao = []
        for word in words:
            max_similarity = 0
            most_similar_word = None
            for vocab_word in vocab:
                try:
                    word_vector = model.wv[word]
                    vocab_word_vector = model.wv[vocab_word]
                    dao_similarity = self.dao_similarity(word, vocab_word)
                    if dao_similarity > max_similarity:
                        max_similarity = dao_similarity
                        most_similar_word = vocab_word
                except KeyError:
                    pass
            if most_similar_word:
                similar_words_with_dao.append((most_similar_word, max_similarity))
        return similar_words_with_dao

    def dao_similarity(self, word1, word2):
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        if not synsets1 or not synsets2:
            return 0
        max_similarity = 0
        for synset1 in synsets1:
            for synset2 in synsets2:
                distance = synset1.shortest_path_distance(synset2)
                if distance is not None and distance > 0:
                    similarity = 1 / distance
                    if similarity > max_similarity:
                        max_similarity = similarity
        return max_similarity
    
    def hitung_probability(self, reduced_query, file_ontologi, classes):
        onto = ontologi()
        score_doc = []

        # For Data Science and Pattern Recognition (DSPR):
        list_of_vocab_DSPR = onto.sparqlQueryDSPR(reduced_query, file_ontologi)
        preFileDSPR = Preprocessing(list_of_vocab_DSPR)
        unique_words_DSPR = preFileDSPR.DoPreProcess(list_of_vocab_DSPR)
        reduced_query_DSPR = query_reduction.reduce_query(unique_words_DSPR)

        similar_words_with_dao_DSPR = self.fasttext_with_dao_similarity(reduced_query, reduced_query_DSPR)
        similar_words_with_dao_DSPR.sort(key=lambda x: x[1], reverse=True)

        probability_scores_DSPR = [score for _, score in similar_words_with_dao_DSPR]
        if probability_scores_DSPR:
            average_probability_DSPR = sum(probability_scores_DSPR) / len(probability_scores_DSPR)
            average_probability_DSPR = round(average_probability_DSPR, 4)
            score_doc.append((1, average_probability_DSPR))  # Class 1 for DSPR
            print("Dao Similarity for Data Science and Pattern Recognition:", average_probability_DSPR)
        else:
            print("No probability scores found for DSPR.")

        # For Distributed Systems (DS):
        list_of_vocab_DS = onto.sparqlQueryDS(reduced_query, file_ontologi)
        preFileDS = Preprocessing(list_of_vocab_DS)
        unique_words_DS = preFileDS.DoPreProcess(list_of_vocab_DS)
        reduced_query_DS = query_reduction.reduce_query(unique_words_DS)

        similar_words_with_dao_DS = self.fasttext_with_dao_similarity(reduced_query, reduced_query_DS)
        similar_words_with_dao_DS.sort(key=lambda x: x[1], reverse=True)

        probability_scores_DS = [score for _, score in similar_words_with_dao_DS]
        if probability_scores_DS:
            average_probability_DS = sum(probability_scores_DS) / len(probability_scores_DS)
            average_probability_DS = round(average_probability_DS, 4)
            score_doc.append((2, average_probability_DS))  # Class 2 for DS
            print("Average Dao Similarity for Distributed Systems:", average_probability_DS)
        else:
            print("No probability scores found for DS.")

        # For Graphics and Visualization (GV):
        list_of_vocab_GV = onto.sparqlQueryGV(reduced_query, file_ontologi)
        preFileGV = Preprocessing(list_of_vocab_GV)
        unique_words_GV = preFileGV.DoPreProcess(list_of_vocab_GV)
        reduced_query_GV = query_reduction.reduce_query(unique_words_GV)

        similar_words_with_dao_GV = self.fasttext_with_dao_similarity(reduced_query, reduced_query_GV)
        similar_words_with_dao_GV.sort(key=lambda x: x[1], reverse=True)

        probability_scores_GV = [score for _, score in similar_words_with_dao_GV]
        if probability_scores_GV:
            average_probability_GV = sum(probability_scores_GV) / len(probability_scores_GV)
            average_probability_GV = round(average_probability_GV, 4)
            score_doc.append((3, average_probability_GV))  # Class 4 for GV
            print("Average Dao Similarity for Graphics and Visualization:", average_probability_GV)
        else:
            print("No probability scores found for GV.")

        # For Natural Language Processing (NLP):
        list_of_vocab_NLP = onto.sparqlQueryNLP(reduced_query, file_ontologi)
        preFileNLP = Preprocessing(list_of_vocab_NLP)
        unique_words_NLP = preFileNLP.DoPreProcess(list_of_vocab_NLP)
        reduced_query_NLP = query_reduction.reduce_query(unique_words_NLP)

        similar_words_with_dao_NLP = self.fasttext_with_dao_similarity(reduced_query, reduced_query_NLP)
        similar_words_with_dao_NLP.sort(key=lambda x: x[1], reverse=True)

        probability_scores_NLP = [score for _, score in similar_words_with_dao_NLP]
        if probability_scores_NLP:
            average_probability_NLP = sum(probability_scores_NLP) / len(probability_scores_NLP)
            average_probability_NLP = round(average_probability_NLP, 4)
            score_doc.append((4, average_probability_NLP))  # Class 3 for NLP
            print("Average Dao Similarity for Natural Language Processing:", average_probability_NLP)
        else:
            print("No probability scores found for NLP.")

        return score_doc

# Preprocessing class
class Preprocessing:
    def __init__(self, judul):
        self.judul = judul

    def toStopwordRemoval(self, kata_kata):
        if isinstance(kata_kata, list):
            kata_kata = ' '.join(kata_kata)
        words = word_tokenize(kata_kata.lower())
        stop_words = set(stopwords.words('indonesian'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        return filtered_words

    def lemmatize_word(self, words):
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return lemmatized_words

    def DoPreProcess(self, doc):
        stopwordFile = self.toStopwordRemoval(doc)
        lemmaFile = self.lemmatize_word(stopwordFile)
        return lemmaFile

class ontologi:
  def sparqlQueryDSPR(self,words, file_ontologi):

      onto = get_ontology(file_ontologi).load()
      graph = default_world.as_rdflib_graph()
      list_of_sentence = []

      # Convert the list of words to a string
      #word_str = '|'.join(word)

      #print(word_str)
      for wrd in words:
        #print (wrd)
        result = list(default_world.sparql("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX owl: <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#>
        SELECT distinct ?subclass ?subclassLabel
        WHERE {
          ?subclass rdfs:subClassOf* <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#Data_Science_and_Pattern_Recognition> .
          ?subclass rdfs:label ?subclassLabel .
          FILTER regex(?subclassLabel, "%s", "i") .
        }


        """ % wrd))

        #print("result : ", result)

        if len(result) != 0:
          for res in result:
            list_of_sentence.append(res[1])

          #print("list of sentence : ", list_of_sentence)

      return list_of_sentence

  def sparqlQueryDS(self,words, file_ontologi):

      onto = get_ontology(file_ontologi).load()
      graph = default_world.as_rdflib_graph()
      list_of_sentence = []

      # Convert the list of words to a string
      #word_str = '|'.join(word)

      #print(word_str)
      for wrd in words:
        #print (wrd)
        result = list(default_world.sparql("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX owl: <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#>
        SELECT distinct ?subclass ?subclassLabel
        WHERE {
          ?subclass rdfs:subClassOf* <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#Distributed_Systems> .
          ?subclass rdfs:label ?subclassLabel .
          FILTER regex(?subclassLabel, "%s", "i") .
        }


        """ % wrd))

        #print("result : ", result)

        if len(result) != 0:
          for res in result:
            list_of_sentence.append(res[1])
          #print("list of sentence : ", list_of_sentence)


      return list_of_sentence

  def sparqlQueryGV(self,words, file_ontologi):

      onto = get_ontology(file_ontologi).load()
      graph = default_world.as_rdflib_graph()
      list_of_sentence = []

      # Convert the list of words to a string
      #word_str = '|'.join(word)

      #print(word_str)
      for wrd in words:
        #print (wrd)
        result = list(default_world.sparql("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX owl: <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#>
        SELECT distinct ?subclass ?subclassLabel
        WHERE {
          ?subclass rdfs:subClassOf* <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#Graphics_and_visualization> .
          ?subclass rdfs:label ?subclassLabel .
          FILTER regex(?subclassLabel, "%s", "i") .
        }


        """ % wrd))

        #print("result : ", result)

        if len(result) != 0:
          for res in result:
            list_of_sentence.append(res[1])
          #print("list of sentence : ", list_of_sentence)


      return list_of_sentence

  def sparqlQueryNLP(self,words, file_ontologi):

      onto = get_ontology(file_ontologi).load()
      graph = default_world.as_rdflib_graph()
      list_of_sentence = []

      # Convert the list of words to a string
      #word_str = '|'.join(word)

      #print(word_str)
      for wrd in words:
        #print (wrd)
        result = list(default_world.sparql("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX owl: <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#>
        SELECT distinct ?subclass ?subclassLabel
        WHERE {
          ?subclass rdfs:subClassOf* <http://www.semanticweb.org/user/ontologies/2024/2/untitled-ontology-96#Natural_Language_Processing> .
          ?subclass rdfs:label ?subclassLabel .
          FILTER regex(?subclassLabel, "%s", "i") .
        }


        """ % wrd))

        #print("result : ", result)

        if len(result) != 0:
            for res in result:
              list_of_sentence.append(res[1])


          #print("list of sentence : ", list_of_sentence)
      return list_of_sentence

class QueryReduction:
    def __init__(self):
        pass

    def get_synonyms_wordnet(self, word):
        synonyms = []
        for syn in wn.synsets(word, lang='ind'):
            for lemma in syn.lemmas(lang='ind'):
                synonyms.append(lemma.name())
        return list(set(synonyms))

    def reduce_query(self, query_words):
        reduced_query = []
        for word in query_words:
            synonyms = self.get_synonyms_wordnet(word)
            if synonyms:
                reduced_query.append(word)
                reduced_query.append(synonyms[0]) #Adds an indented block with a statement to execute when the condition is true
            else:
                reduced_query.append(word)

        #print("sebelum : ",reduced_query)
        reduced_query = self.remove_duplicate_words(reduced_query)
        #print("sesudah : ",reduced_query)
        return reduced_query

    def remove_duplicate_words(self,words):
      """
      Removes duplicate words from a list of words.

      Args:
        words: A list of words.

      Returns:
        A new list with duplicate words removed.
      """
      return list(dict.fromkeys(words))
    
no = 0
data_to_save = []
query_reduction = QueryReduction()

import streamlit as st

st.title("Ontology-Based Classification System")

# File upload for ontology
uploaded_file = st.file_uploader("Upload Ontology File", type=["owl"])

if uploaded_file:
    st.success("Ontology file uploaded successfully!")
    file_ontologi = uploaded_file.name

# Input text from user for query
user_query = st.text_area("Enter the query text:", "")

if st.button("Classify"):
    if uploaded_file and user_query:
        # Process the query
        preprocessor = Preprocessing(user_query)
        processed_query = preprocessor.DoPreProcess(user_query.split())

        pembobotan = Pembobotan()
        ontology_file = file_ontologi

        # Call your function to classify
        scores = pembobotan.hitung_probability(processed_query, ontology_file, classes=None)

        # Sort scores by probability in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

        
        # Mapping for class labels
        class_labels = {
            1: "Data Science and Pattern Recognition",
            2: "Distributed Systems",
            3: "Graphics and Visualization",
            4: "Natural Language Processing"
        }

        # Display all classification scores
        st.write("### Classification Results:")
        for class_id, score in scores:
            st.write(f"Class {class_id} ({class_labels[class_id]}): Score = {score:.4f}")

        # Find the class with the highest score
        if scores:
            highest_score_class = scores[0]  # Get the first element which has the highest score
            class_id, score = highest_score_class

            # Display the highest classification result
            st.write(f"#### Hasil Klasifikasi: {class_labels[class_id]}")
        else:
            st.write("No classification scores available.")
    else:
        st.error("Please upload an ontology file and enter a query.")


