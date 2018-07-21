import csv

names = []

# read in the tsv file to gather all movie or TV show titles
with open("stage1_docs/Data/name.basics.tsv", 'r') as file:
    rd = csv.reader(file, delimiter="\t", quotechar='"')
    for row in rd:
        names.append(str(row[1]))

# write out all the titles to a file
with open("stage1_docs/Data/actor_names.csv", 'w') as file:
    for i in range(0, len(names)):
        file.write(names[i])
        if i < len(names)-1:
            file.write('\n')