import csv

titles = []

# read in the tsv file to gather all movie or TV show titles
with open("stage1_docs/Data/title.basics.tsv", 'r') as file:
    rd = csv.reader(file, delimiter="\t", quotechar='"')
    for row in rd:
        if row[1] in ['movie', 'short', 'tvSeries']:
            titles.append(str(row[2]))

# write out all the titles to a file
with open("stage1_docs/Data/titles.csv", 'w') as file:
    for title in titles:
        file.write(title)
        if title != titles[-1]:
            file.write('\n')