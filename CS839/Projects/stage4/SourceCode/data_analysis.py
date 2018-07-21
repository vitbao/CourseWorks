import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

# specify data location
DATA = 'E.csv'
FOLDER = './Data/'

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

def profile_data(column):
    # examine a dataframe column of numeric values for missing values and range
    countNAN = 0
    values = []
    for item in column:
        if pd.isnull(item):
            countNAN += 1
        else:
            if column.name == 'budget' or column.name == 'revenue' or column.name == 'opening_weekend_revenue':
                symbol, value = extract_symbol(item)
                values.append(float(value))
            elif column.name == 'release_year':
                values.append(int(item))
            else:
                values.append(float(item))
    min_value = np.min(values)
    max_value = np.max(values)
    mean = np.average(values)
    stdev = np.std(values)        
    return (min_value, max_value, mean, stdev, countNAN)
        
def extract_data(df):
    ''' because there are large variances in budget and revenue due to release_year and currency
    (not all values are in US Dollars), let's extract movies that have budget and revenue in US Dollars only,
    and are released within the last 20 years'''
    # extract revenue, budget, rating for each release year.
    # extract genres for each release_year
    # only extract tuples that have the budget and revenue of US currency symbol ($)
    year_boxOffice = {}
    year_genres = {}
    
    for index, movie in df.iterrows():
        if pd.isnull(movie['budget']) or pd.isnull(movie['revenue']):
            continue
        else:
            budget_symbol, budget_value = extract_symbol(movie['budget'])
            revenue_symbol, revenue_value = extract_symbol(movie['revenue'])
            release_year = int(movie['release_year'])
            rating = float(movie['rating'])
            if budget_symbol == revenue_symbol and budget_symbol == '$' and release_year >= 1998 and release_year < 2018:
                if release_year not in year_boxOffice.keys():
                    year_boxOffice[release_year] = [[float(budget_value), float(revenue_value), rating]]
                    year_genres[release_year] = [movie['genres']]
                else:
                    year_boxOffice[release_year].append([float(budget_value), float(revenue_value), rating])
                    year_genres[release_year].append(movie['genres'])
            else:
                continue
    # sort data by keys:
    sorted_keys = sorted(year_boxOffice.keys())
    sorted_data = {}
    sorted_year_genres = {}
    for key in sorted_keys:
        sorted_data[key] = year_boxOffice[key]
        sorted_year_genres[key] = year_genres[key]
                
    return sorted_data, sorted_year_genres

def summarize_data(data):
    # computes average and standard deviation
    summarized_data = {}
    for key in data.keys():
        values = np.asarray(data[key])
        summarized_data[key] = [np.average(values[:, 0]), np.average(values[:, 1]), np.std(values[:, 0]), np.std(values[:, 1])]
    return summarized_data
    
def make_scatter_plot(data):
    fig, ax = plt.subplots()
    index = 0
    for key in data.keys():
        if key >= 2008:
            value = np.asarray(data[key])
            budget = value[:,0]
            revenue = value[:, 1]
            ax.scatter(budget, revenue, label = key)
        else:
            continue
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(loc = 'best', ncol = 2, labelspacing = 0.5, fontsize = 8)
    plt.xlabel('Movie Budget ($)')
    plt.ylabel('Movie Revenue ($)')
    plt.show()
    plt.savefig('ScatterPlot.png')
    plt.close(fig)

def make_bar_graph(data):
    mean_budget = []
    stdev_budget = []
    mean_revenue = []
    stdev_revenue = []
    for key in data.keys():
        mean_budget.append(data[key][0])
        stdev_budget.append(data[key][1])
        mean_revenue.append(data[key][2])
        stdev_revenue.append(data[key][3])
    fig, ax = plt.subplots()
    x_pos = np.arange(len(data.keys()))
    bar_width = 0.2
    budget = ax.bar(x_pos - bar_width/2, mean_budget, bar_width,
                color='Blue', label='Avg_Budget', align = 'edge', yerr = stdev_budget)
    revenue = ax.bar(x_pos + bar_width/2, mean_revenue, bar_width,
                color='Orange', label='Avg_Revenue', align = 'edge',yerr = stdev_budget)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('US Dollars')
    plt.xlabel('years')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(data.keys(), fontsize = 10, rotation = 45)
    plt.legend()
    plt.show()
    plt.savefig('Bargraph.png')
    plt.close(fig)
    
def compute_corrCoeff(data):
    budget = []
    revenue = []
    for key in data.keys():
        value = np.asarray(data[key])
        budget.extend(value[:,0])
        revenue.extend(value[:, 1])
    return np.corrcoef(budget, revenue)[0, 1]

def drill_down(df, year):
    # drill down into movies of a particular year
    max_value = 0
    max_value_id = 0
    for index, row in df.iterrows():
        if int(row['release_year']) == year:
            if pd.isnull(row['budget']) or pd.isnull(row['revenue']):
                continue
            else:
                symbol_budget, budget_value = extract_symbol(row['budget'])
                symbol_revenue, revenue_value = extract_symbol(row['revenue'])
                if symbol_budget == symbol_revenue and symbol_budget == '$':
                    if float(revenue_value) > max_value:
                        max_value = float(revenue_value)
                        max_value_id = index
    #print(max_value)
    return df.iloc[max_value_id]
                
def main():
    df = pd.read_csv(FOLDER + DATA)
    # profile budget, revenue, release_year, rating
    attributes = ['budget', 'revenue', 'release_year', 'rating']
    for attribute in attributes:
        column = df[attribute]
        stats = profile_data(column)
        print('statistics of attribute {}: min = {}, max = {}, mean = {},\
              stdev = {}, number of missing values = {}'.format(attribute, stats[0], stats[1], stats[2], stats[3], stats[4]))
    # budget and revenue have large range and hence large variances, let's extract a subset of data for analysis
    data, year_genres_dict = extract_data(df)
    # summarize data by mean and visualize it using bar-graphs
    summarized_data = summarize_data(data)
    make_bar_graph(summarized_data)
    # visualize individual movies' budget and revenue for the past 10 years (excluding 2018)
    make_scatter_plot(data)
    # compute Pearson correlation coefficient
    corr_coeff = compute_corrCoeff(data)
    print(corr_coeff)
    # identify movie that corresponds to highest revenue from the visualization (orange point upperright corner)
    # drill down into movies into 2009 and identify the movies with higest revenue
    movie = drill_down(df, 2009)
    print(movie)
    
    
if __name__ == '__main__':
    main()
                
            
