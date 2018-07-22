# Data analysis
## Input Files for merger.py
The following input files have to be in the Data folder of the current directory: 
- A.csv: The IMDb movies table, contains 3500 tuples (See table's schema below)
- B.csv: The TMDb movies table, contains 5490 tuples (See table's schema below)
- C.csv: Candidate set of all tuple pairs that survive the blocking step, contains 2547 tuples
- G.csv: Labeled sample set of candidate tuples, contains 400 tuples
## Output Files from merger.py
The following output files are saved in Data folder when merger.py is executed. 
- E.csv: The merged table, contains 1111 tuples (See table's schema below)
- ABmatches.csv: The set of matches (ID pairs) between Tables A and B, contains 1111 tuples
## CSV Tables Attributes' Description
Each tuple in table A, B, and E consists of 18 attributes listed below. Attributes 15-18 are unique to IMDb movies. If an attribute has multiple values, the values are separated by semicolons.
1. id: unique id of the movie
2. title: title of the movie
3. cast:	names of the top 5 billed actors/actresses
4. directors: names of directors
5. writers: names of screenplay/novel writers
6. genres:	movie genres (eg. action, adventure, thriller)
7. keywords:	keywords from movie’s plot
8. content_rating:	movie’s certification (eg. R, PG, PG-13)
9. run_time: duration of movie (in minutes)
10. release_year:	the year in which the movie is released
11. languages: the languages that appear in the movie
12. rating:	users’ rating score
13. budget:	movie’s budget
14. revenue: movies’ revenue
15. opening_weekend_revenues: the revenue of the movie in the opening weekend
16. production_companies: the companies that produce the movie
17. production_countries:	the countries that produce the movie
18. alternative_titles:	other titles of the movie


