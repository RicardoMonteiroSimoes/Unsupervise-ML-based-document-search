# IE1-SKB

## Konzept
Die Idee ist es mittels Algorithmen aus dem Bereich des MLs die Suche zu vollziehen. Unter anderem gäbe es als Beispiel das Tf-Idf Mass sowie aus dem Unsupervised learning die automatische Gruppierung ähnlicher Texte. Da wir bereits KI1+2 an der ZHAW hatten, können wir das erlernte mittels diesem Forschungsexperiment in die Praxis umzusetzen. Als Programmiersprachen würde Python genommen und diverse Libraries (skcikit-learn, tensorflow, pytorch)

Der Umsetzungsplan wäre wie folgt:
- Erhaltene Datenstruktur verarbeiten
- Datenstruktur mittels ML-Algorithmen analysieren und statisch hinterlegen
- Inhalte eines Suchausdruckes mittels statischen tabellen konvertieren
- Daten mittels Relevanz ordnen.

Die grösste Herausforderung ist es, eine statische verlinkung der Daten sowie der ML-Algorithmen zu erstellen, wodurch diese dann mittels Suchanfragen durchsucht werden können.


# Lösung

Die Suchlösung ist auffindbar in den Dateien `tfidf_vectorizer.trec_eval ` und `count_vectorizer.trec_eval `. Diese wurden mit dem script `create_and_score.py` erstellt, das wie folgt funktioniert:

```
create_and_score.py -h
Document Search Engine with unsupervised ML methods
---------------------------------------------------
usage: create_and_score.py [-h] [-t TOPICS] [-mf MAXFEATURES] [-r RANDOMSTATE]
                           [-f FILE] [-d] [-idf] [-s]

Create a document search engine using unsupervised learning

optional arguments:
  -h, --help            show this help message and exit
  -t TOPICS, --topics TOPICS
                        Set how many topics can be categorized
  -mf MAXFEATURES, --maxfeatures MAXFEATURES
                        Sets max features for the vectorizer
  -r RANDOMSTATE, --randomstate RANDOMSTATE
                        Sets a given random state for LDA
  -f FILE, --file FILE  Name the trac_eval file
  -d, --debug           Turn on debug messages
  -idf, --tfidf         Select TFIDF vectorizer instead of count vectorizer
  -s, --stopwords       Sets english for stopwords, default is none
```

Die Dateien `ie1_collection.trec` und `ie1_queries.trec` müssen im Unterordner `data` sein. 

Die `*.trec_eval` Dateien wurden mit den Standardeinstellungen gebaut. Jeweils nur die `-idf` option wurde genommen für die `TfIdf` Variante.
