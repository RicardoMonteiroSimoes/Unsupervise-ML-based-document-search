# IE1-SKB

## Konzept
Unsere Idee ist es das Tf-Idf Mass zu verwenden, wobei die Suchanfragen die Terme(T) sind und die Suchergebnisse die Dokumente(D). Da wir im vorherigen Semester im KI2 Modul dieses Konzept kennengelernt haben, ist dieses Forschungsexperiment eine ideale Möglichkeit das Gelernte in die Praxis umzusetzen. Wir würden für die Realisierung bzw. Forschung ein Python Programm schreiben und dafür geeignete Libraries verwenden (z.B. scikit-learn).

Im erschten Schritt werden die Dokumente nach der Häufigkeit des Auftretens der Worte für eine Suchanfrage T sortiert (term frequency). Im zweiten Schritt werden die Worte der Suchanfrage T gewichtet, um somit z.B häufig auftretende Wörter weniger stark zu gewichten (inverse document frequency).
