import py_entitymatching as em
import pandas as pd
import numpy as np
import os, sys
import py_stringmatching as sm
#import pandas_profiling

# Specify csv files and directory (if any)
Directory = './Data/'
IMDb = 'imdb_movies.csv'
TMDb = 'themoviedb.csv'

# clean up imdb_movies.csv and themoviedb.csv
def clean_up():
    A = pd.read_csv(Directory+IMDb)
    B = pd.read_csv(Directory+TMDb)
    A_size = A.shape
    B_size = B.shape
    # replace "not available" in TMDb by empty string
    B.replace("not available", "", inplace = True)
    A_ids = ["a%d" %i for i in range(0, A_size[0])]
    B_ids = ["b%d" %i for i in range(0, B_size[0])]
    A.insert(0, 'id', A_ids)
    B.insert(0, 'id', B_ids)
    A.to_csv(Directory+'A.csv', index = False)
    B.to_csv(Directory+'B.csv', index = False)

def blocking_rules(x, y):
    # return True if x and y survive the blocking rules
    # x and y are pandas series
    x_directors = str(x['directors']).split(';')
    y_directors = str(y['directors']).split(';')
    
    x_writers = str(x['writers']).split(';')
    y_writers = str(y['writers']).split(';')
    x_actors = str(x['cast']).split(';')
    y_actors = str(y['cast']).split(';')
    director_match = False
    writer_match = False
    actor_match = False
    overlap_size = 0
    # create a tokenizer
    ws_tok = sm.WhitespaceTokenizer()
    # create a Jaccard similarity measure object
    jac = sm.Jaccard()
    for x_director in x_directors:
        if director_match == True:
            break
        else:
            # tokenize x_director using whitespace
            if x_director == 'nan':
                continue
            else:
                x_director = ws_tok.tokenize(x_director)
                for y_director in y_directors:
                    if y_director == 'nan':
                        continue
                    else:
                        # tokenize y_director using whitespace
                        y_director = ws_tok.tokenize(y_director)
                        if jac.get_sim_score(x_director, y_director) >= 0.8:
                            director_match == True
                            break
    for x_writer in x_writers:
        if writer_match == True:
            break
        else:
            if x_writer == 'nan':
                continue    
            else:
                x_writer = ws_tok.tokenize(x_writer)
                for y_writer in y_writers:
                    if y_writer == 'nan':
                        continue
                    else:
                        y_writer = ws_tok.tokenize(y_writer)
                        if jac.get_sim_score(x_writer, y_writer) >= 0.8:
                            writer_match = True
                            break
    for x_actor in x_actors:
        if actor_match == True:
            break
        else:
            if x_actor == 'nan':
                continue
            else:
                x_actor = ws_tok.tokenize(x_actor)
                for y_actor in y_actors:
                    if y_actor == 'nan':
                        continue
                    else:
                        y_actor = ws_tok.tokenize(y_actor)
                        if jac.get_sim_score(x_actor, y_actor) >= 0.8:
                            actor_match = True
                            break
    if actor_match == False and director_match == False and writer_match == False:
        return True
    else:
        return False    
    
def black_box_blocker(A, B): 
    # return candidate set of tuples surving blocking rules
    ab = em.AttrEquivalenceBlocker()
    # block on release_year
    
    C1 = ab.block_tables(A, B, l_block_attr='release_year', r_block_attr='release_year', 
                    l_output_attrs=list(A),
                    r_output_attrs=list(B),
                    l_output_prefix='l_', r_output_prefix='r_')
    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(blocking_rules)
    C2 = bb.block_candset(C1, show_progress=False )
    return C2
    
def blocker_debugging(C, A, B):
    # returned the tuples that are thrown away by the blocker
    dbq = em.debug_blocker(C, A, B, output_size = 100)
    return dbq

def main():
    # clean up original csv
    # comment out this code if A.csv and B.csv already exists
    #clean_up()
    # read csv tables
    A = pd.read_csv(Directory+'A.csv')
    B = pd.read_csv(Directory+'B.csv')
    # set keys to tables
    em.set_key(A, 'id')
    em.set_key(B, 'id')
    # block tables using black-box blocker
    C = black_box_blocker(A, B)
    C.to_csv('C.csv', index = False)
    # debug blocker
    dbq = blocker_debugging(C, A, B)
    dbq.to_csv('debugged_result.csv')
    
if __name__ == '__main__':
    main()
