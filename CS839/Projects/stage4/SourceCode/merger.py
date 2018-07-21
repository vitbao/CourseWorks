# Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd
import py_stringmatching as sm
import numpy as np

# Specify directory 
FOLDER = './Data/'

# Set seed value (to get reproducible results)
seed = 0

def read_files():
    # Read in data files
    A = em.read_csv_metadata(FOLDER+'A.csv', key = 'id') # imdb data
    B = em.read_csv_metadata(FOLDER+'B.csv', key = 'id') # tmdb data
    C = em.read_csv_metadata(FOLDER+'C.csv', key = '_id', ltable = A, rtable = B,
                             fk_ltable = 'l_id', fk_rtable = 'r_id') # candidates that survive blocking step
    G = em.read_csv_metadata(FOLDER+'G.csv', key = '_id', ltable = A, rtable = B,
                             fk_ltable = 'l_id', fk_rtable = 'r_id') # labeled data
    
    return A, B, C, G

def predict_matching_tuples(A, B, C, G):
    # Split G into I and J for CV
    IJ = em.split_train_test(G, train_proportion = 0.5, random_state = 0)
    I = IJ['train']
    # Generate features set F
    F = em.get_features_for_matching(A, B, validate_inferred_attr_types = False)
    # Convert G to a set of feature vectors using F
    H = em.extract_feature_vecs(I, feature_table = F, attrs_after = 'label',
                                show_progress = False)
    excluded_attributes = ['_id', 'l_id', 'r_id', 'label']
    # Fill in missing values with column's average
    H = em.impute_table(H, exclude_attrs = excluded_attributes,
                        strategy='mean')
    # Create and train a logistic regression - the best matcher from stage3.
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    lg.fit(table = H, exclude_attrs = excluded_attributes, target_attr = 'label')
    # Convert C into a set of features using F
    L = em.extract_feature_vecs(C, feature_table = F, show_progress = False)
    # Fill in missing values with column's average
    L = em.impute_table(L, exclude_attrs=['_id', 'l_id', 'r_id'], 
                        strategy='mean')
    # Predict on L with trained matcher
    predictions = lg.predict(table = L, 
                             exclude_attrs=['_id', 'l_id', 'r_id'], 
                             append = True, target_attr = 'predicted',
                             inplace = False, return_probs = False, 
                             probs_attr = 'proba')
    # Extract the matched pairs' ids 
    matched_pairs = predictions[predictions.predicted==1]
    matched_ids = matched_pairs[['l_id', 'r_id']]
    # Save matched_pairs to file so we don't have to train and predict each time the code is executed
    matched_ids.to_csv(FOLDER + 'predictedMatchedIDs.csv', index = False)

def resolve_mismatches(df):
    # remove manually inspected mismatches
    # Inspect 'l_id' and 'r_id' for potential mismatches. 
    l_dup = df[df.duplicated(subset = 'l_id', keep = False)]
    r_dup = df[df.duplicated(subset = 'r_id', keep = False)]
    # Uncomment out the print statements below to view duplicates
    #print(l_dup)
    #print(r_dup)
    # After manually inspecting the duplicates, found the following mismatches:
    mismatches = [['a872', 'b2879'], ['a987', 'b3508'], ['a2730', 'b4185'], ['a1723', 'b386'], ['a1843', 'b223']]
    # Make a copy of matched_ids dataframe
    matchedIDs = df.copy()
    # Find indices of mismatched rows
    mismatched_indices = []
    for index, row in df.iterrows():
        for mismatch in mismatches:
            if row['l_id'] == mismatch[0] and row['r_id'] == mismatch[1]:
                mismatched_indices.append(index)
    #print(mismatched_indices)
    # Drop the mismatched rows from the matched table
    matchedIDs.drop(matchedIDs.index[mismatched_indices], inplace = True)

    return matchedIDs

def merge_tables(A, B, matchedIDs):
    # merge tables A and B into table E using matched IDs
    # first create a new data frame E with schema similar to A and B
    merged_data = []
    idx = 0
    for index, row in matchedIDs.iterrows():
        tupleA = A[A['id'] == row['l_id']]
        tupleB = B[B['id'] == row['r_id']]
        tupleE = merge_tuples(tupleA, tupleB, idx)
        merged_data.append(tupleE)
        idx = idx + 1
    E = pd.DataFrame(data = merged_data, columns = list(A))
    E.to_csv(FOLDER+'E.csv', index = False)
        
def merge_tuples(tupleA, tupleB, index):
    # merging two tuples A, B into tuple E
    attributes = list(tupleA)
    revenue_budget_attributes = ['budget', 'revenue', 'opening_weekend_revenue']
    multiple_value_attributes = ['directors', 'writers', 'cast', 'genres', 'keywords',
                                 'languages', 'production_companies','production_countries']
    merged_values = []
    for attribute in attributes:
        valueA = tupleA[attribute].values[0]
        valueB = tupleB[attribute].values[0]
        if attribute == 'id':
            merged_value = index
        elif attribute == 'title':
            merged_value = valueA
        elif attribute == 'content_rating':
            if pd.isnull(valueA):
                merged_value = valueB
            else:
                merged_value = valueA
        elif attribute == 'run_time':
            if pd.isnull(valueA):
                merged_value = valueB
            elif pd.isnull(valueB):
                merged_value = valueA
            else:
                merged_value = int(round((float(valueA) + float(valueB))/2))
        elif attribute == 'rating':
            if pd.isnull(valueA):
                merged_value = float(valueB)/10
            elif pd.isnull(valueB):
                merged_value = valueA
            else:
                merged_value = round((float(valueA) + float(valueB)/10)/2,1)
        elif attribute == 'release_year':
            merged_value = valueA
        elif attribute in revenue_budget_attributes:
            merged_value = merge_money(valueA, valueB)
        elif attribute in multiple_value_attributes:
            merged_value = merge_multiple_values_attribute(valueA, valueB)
        else:
            merged_value = merge_alternative_title(tupleA, tupleB)
        merged_values.append(merged_value)
        
    return merged_values
        
def merge_money(a, b):
    if pd.isnull(a):
        if not pd.isnull(b):
            symbolb, numberb = extract_symbol(b)
            if symbolb == '-':
                symbolb = ''
            return symbolb + numberb
        else:
            return a
    elif pd.isnull(b):
        if not pd.isnull(a):
            symbola, numbera = extract_symbol(a)
            if symbola == '-':
                symbolb = ''
            return symbola + numbera
        else:
            return a
    else:
        symbola, numbera = extract_symbol(a)
        symbolb, numberb = extract_symbol(b)
        if symbola == '-':
            symbola = ''
        if symbolb == '-':
            symbolb = ''
        if symbola != symbolb:
            if symbola == '$':
                return symbola + numbera
            elif symbolb == '$':
                return symbolb + numberb
            else:
                return symbola + numbera
        else:
            merged_value = int(round((float(numbera) + float(numberb))/2))
            return symbola + str(merged_value)

def extract_symbol(x):
    # extract the currency symbol from string x
    symbol = ''
    number = ''
    for i in range(0, len(x)):
        if not x[i].isdigit():
            symbol = symbol + x[i]
        else:
            number = number + x[i:]
            break
    return symbol, number

def merge_multiple_values_attribute(a, b):
    """Returned the merged content of the two cell a and b.
    a and be must be strings likes directors, writers, etc.
    We assume ';' as delimeters.
    Allow partial matching for string names using Jaccard score threshold at 0.8"""
    if (pd.isnull(a)):
        return b
    elif (pd.isnull(b)):
        return a
    else:
        a_list = list(set(a.split(';')))
        b_list = list(set(b.split(';')))
        merged_value = []
        for string_a in a_list:
            for string_b in b_list:
                score = compute_simScore(string_a, string_b)
                if score >= 0.6:
                    b_list.remove(string_b)
                else:
                    continue
        merged_value.extend(a_list)
        merged_value.extend(b_list)
        merged_value = ';'.join(merged_value)

        return merged_value

def compute_simScore(str1, str2):
    # compute similarity score between str1 and str2 using Smith Waterman measure
    sw = sm.SmithWaterman()
    return sw.get_raw_score(str1, str2)/min(len(str1), len(str2))    

def merge_alternative_title(tupleA, tupleB):
    # merge alternative title
    altTitles = tupleA['alternative_titles'].values[0]
    altTitles = altTitles.split(';')
    titleA = tupleA['title'].values[0]
    titleB = tupleB['title'].values[0]
    simScore = compute_simScore(titleA, titleB)
    match = False
    # if titleB does not match titleA, considering add it to altTitles 
    if simScore < 0.8:
        for title in altTitles:
            score = compute_simScore(title, titleB)
            if score >= 0.8:
                match = True
                break
        if match == False:
            # add titleB to altTitles if it does not match any of the alternative titles
            altTitles.append(titleB)
    
    merged_value = ';'.join(altTitles)

    return merged_value

def main():
    # read files
    A, B, C, G = read_files()
    # predict matching tuples if matchedIDs.csv is not in the Data folder.
    if not os.path.exists(FOLDER + 'predictedMatchedIDs.csv'):
        predict_matching_tuples(A, B, C, G)
    predicted_matchedIDs = pd.read_csv(FOLDER + 'predictedMatchedIDs.csv')
    # resolve any mismatches in the matched table
    matchedIDs = resolve_mismatches(predicted_matchedIDs)
    matchedIDs.to_csv(FOLDER+'ABmatches.csv', index = False)
    merge_tables(A, B, matchedIDs)

if __name__ == '__main__':
    main()
