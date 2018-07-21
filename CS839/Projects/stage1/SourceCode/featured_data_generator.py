# -*- coding: utf-8 -*-
__author__ = 'Trang Vu'
'''
    inputs: marked up '.txt' files
    outputs: a csv file with each entry correspond to a string, which has the following fields:
        string id: integer
        the string of either 1, 2, 3, or 4 consecutive words: string 
        file id/file name: string 
        start position of the string: integer
        end position of the string: integer
        ## I don't think we need capitalized anymore since we get rid of non-capitalized ones
        otherEntity: is capitalized but is probably a location/other entity
        near_capitalized: words before or after it are capitalized
        name_suffix: the word ends with a common name suffix, like Jr. or Sr.
        common_word: the word string contains a common stop word (e.g. The, A, etc)
        first_name: at least one word in the string matches a first name dictionary
        actor_legislator_name: at least one word in the string matches at least one part of an actor's name (first or last name)
        class label: binary with 1  or 0 being the string is or is not a person name, respectively 
'''        

# load packages
import os
import string
import re
import pandas as pd
import collections
import nltk
import csv
import random

# list directory paths        
MarkedUp = 'stage1_docs/Data/MarkedUp/'
CleanedMarkedUp = 'stage1_docs/Data/Cleaned_MarkedUp/'
DATA = 'stage1_docs/Data/'
Features = 'stage1_docs/Data/Features/'
TrainingSet = DATA + 'trainingSet/'
TestSet = DATA + 'testSet/'

# Global feature names that will be shared between modules
LOCATION_FEATURES = ['documentID', 'start_index', 'end_index']
OTHER_FEATURES = ['frequency', 'prefixed', 'suffixed', 'otherEntity', 'near_capitalized', 'name_suffix',
                  'common_word', 'first_name', 'actor_legislator_name', 'in_name_list', 'near_parentheses', 'near_verb',
                  'near_adj', 'is_proper_noun']
TITLES_DICT = {}
FIRST_NAMES_DICT = {}
ACTOR_NAMES_DICT = {}
LEGISLATORS_NAMES_DICT = {}
TRAIN_CSV = DATA + 'train_data.csv'
TEST_CSV = DATA + 'test_data.csv'
TITLES_CSV = DATA + 'titles.csv'
FIRST_NAMES_CSV = DATA + 'census-derived-all-first.txt'
ACTOR_NAMES1_CSV = DATA + 'actor_names1.csv'
ACTOR_NAMES2_CSV = DATA + 'actor_names2.csv'

LEGISLATORS_CURRENT_CSV = DATA + 'legislators-current.csv'
LEGISLATORS_HISTORICAL_CSV = DATA + 'legislators-historical.csv'

def clean_file(filename):
    # remove extra newlines in file
    # replace weird apostrophes by single quote
    # check if number of end_tags == number of star_tags
    with open(MarkedUp+filename, encoding = 'utf-8') as inputfile:
        oneline_text = ''
        for text in inputfile:
            text = text.rstrip('\n')
            text = text.replace('’', '\'')
            text = text.replace('`', '\'')
            oneline_text = oneline_text + ' ' + text
    start_tag = '<pname>'
    end_tag = '</pname>'
    num_start_tag = len([i.start() for i in re.finditer(start_tag, oneline_text)])
    num_end_tag = len([i.start() for i in re.finditer(end_tag, oneline_text)])
    if num_start_tag != num_end_tag:
        print ('Tags are mismatched! Check markedup file: ' + filename)
    else:
        cleaned_filename = 'cleaned_' + filename
        with open(CleanedMarkedUp+cleaned_filename, 'w', encoding = 'utf-8') as outputfile:
            outputfile.write(oneline_text)
    return oneline_text
        
def data_generator(fileID, filename, text):
    start_tag = '<pname>'
    end_tag = '</pname>'

    # tokenize the document and generate part of speech tags for each sentence
    pos_words = nltk.sent_tokenize(text)
    pos_words = [nltk.word_tokenize(sent) for sent in pos_words]
    pos_words = [nltk.pos_tag(sent) for sent in pos_words]
    # flatten list of lists
    pos_words = [item for sublist in pos_words for item in sublist]

    # split text by space
    words_by_space = text.split()
    # split words further by punctuations
    words = []
    for word in words_by_space:
        new_word = split_string(word)
        if new_word is not None:
            words.extend(new_word)
        else:
            print('Can not split string: ', word)
            print('Check name labels in file: ', filename)

    # text that we remove all <pname> and </pname>.
    # get word frequency
    word_frequency, untagged_words = get_word_frequency(words, start_tag, end_tag)
    # get part of speech of words
    pos_words = match_words(untagged_words, pos_words)

    # generate list of strings of words
    data = []
    string_id = 0
    count = 0
    for index in range(0, len(words)):
        for num_words in range(1, 5):
            if index + num_words <= len(words): 
                word_string = words[index:index+num_words]
                word_string = ' '.join(word_string)

                # find class label
                class_label = findClassLabel(word_string, start_tag, end_tag)

                #print (word_string)
                start = index
                end = index + num_words

                # remove markedup tags
                ws = removeTags(word_string, start_tag, end_tag)

                # get word_string frequency
                frequency = 1
                if ws in word_frequency.keys():
                    frequency = word_frequency[ws]
                # check if preceding word is a special prefix
                prefix = 0
                if start > 0:
                    prefix = checkPrefix(untagged_words[start - 1])
                    if prefix == 0 and start > 1:
                        prefix = checkPrefix(untagged_words[start - 2])
                # check if succeding word is a special suffix
                suffix = 0
                if end < len(untagged_words):
                    suffix = checkSuffix(untagged_words[end])
                    if suffix == 0 and end < len(untagged_words)-1:
                        suffix = checkSuffix(untagged_words[end+1])
                otherEntity = 0
                if start > 0 and end < len(untagged_words):
                    otherEntity = checkOthers(untagged_words, start - 1, end)


                if class_label == 1:
                    count = count + 1
                # is another entity if it's a title
                if otherEntity == 0:
                    otherEntity = is_title(ws)
                # check if words in string all capitalized (or is a word in a special dictionary like 'van' or 'del')
                capitalized = isCapitalized(ws)
                # check if word contains punctuation besides .
                punctuation = contains_punctuation_except_some(ws)
                # check if previous or next word in document is capitalized
                near_capitalized = is_near_capitalized(untagged_words, start, end)
                # checks if the word string ends with a name suffix, like Sr. or Jr.
                name_suffix = is_name_suffix(ws)
                # checks to see if the word string contains a common stop word
                common_word = contains_common_word(ws)
                # checks to see if string contains a first name from the name dictionary
                first_name = contains_first_name(ws)
                # checks to see if string contains part of an actor's or legislator's name from dictionary
                actor_legislator_name = contains_actor_name(ws) + contains_legislator_name(ws)
                # checks to see if name is in a list
                in_name_list = inNamelist(untagged_words, start, end, ws)
                # checks to see if candidate is near parentheses
                near_parentheses = isNearParentheses(untagged_words, start, end, ws)
                # pos checking: check for verbs/adjectives (starts with 'VB' or 'JJ') before/after,
                near_verb = isNearVerb(pos_words, start, end)
                near_adj = isNearAdj(pos_words, start, end)
                # check to see if all the words in a string are NNP (proper noun)
                is_proper_noun = isProperNoun(pos_words, start, end)
                # if word is not capitalized, throw it away: pruning
                if capitalized != 1 or punctuation or is_common_word(ws) or in_blacklist(ws):
                    continue
                frequency = text.count(ws)
                # create data instance
                data_instance = [string_id, ws, filename, fileID, start, end, frequency, prefix, suffix,
                                 otherEntity, near_capitalized, name_suffix, common_word, first_name, actor_legislator_name,
                                 in_name_list, near_parentheses, near_verb, near_adj, is_proper_noun, class_label]
                data.append(data_instance)
                string_id = string_id + 1

    # sanity check: make sure no marked up tags were accidentally thrown out
    if num_of_labels(data) != count:
        print("Error: A label did not make it through! Check file {} for potential errors.".format(filename))
    return data       


def isNearVerb(pos_words, start, end):
    if start > 0:
        if pos_words[start-1][1].startswith('VB'):
            return 1

    if end < len(pos_words):
        if pos_words[end][1].startswith('VB'):
            return 1

    return 0


def isNearAdj(pos_words, start, end):
    if start > 0:
        if pos_words[start-1][1].startswith('JJ'):
            return 1

    if end < len(pos_words):
        if pos_words[end][1].startswith('JJ'):
            return 1

    return 0


def isProperNoun(pos_words, start, end):
    for word in pos_words[start:end]:
        if word[1] != 'NNP':
            return 0

    return 1


def match_words(words, pos):
    '''
    Generates a new list based on the match ups of the words and the part of speech tags.
    Ex: [('blue', 'JJ'), ('running', 'VB')]
    '''
    matched_words = []
    ind = 0
    for word in words:
        flag = 0
        for i in range(ind, len(pos)):
            if word == pos[i][0]:
                matched_words.append((word, pos[i][1]))
                flag = 1
                break
        if flag == 0:
            matched_words.append((word, ''))

    return matched_words


def in_blacklist(word_string):
    organizations = ['Capcom', 'New York Times', 'NFL', 'NASA', 'JPL', 'Malin Space Science Systems',
                     'SpaceX', 'White House', 'Sony', 'Disney', 'Walt Disney', 'Touchstone', 'Oscars',
                     'Sundance', 'Variety', 'Atlantic', 'Instagram', 'Facebook', 'Twitter']
    clubs = ['West Bromwich Albion', 'Manchester City', 'Crystal Palace', 'Sevilla', 'Östersund', 'Arsenal',
             'Liverpool', 'Norwich City', 'Red Star Belgrade', 'St Johnstone', 'Patriots', 'Seahawks',
             'Oakland Raiders', 'Dallas Cowboys', 'New York Giants', 'Eagles']
    locations = ['Seattle', 'Philadelphia', 'New Orleans', 'Minnesota', 'San Francisco', 'Cleveland', 'Arizona',
                 'Denver', 'Pyeongchang', 'Seoul', 'Tokyo', 'Cannes', 'Festival', 'New York',
                 'Washington DC', 'America', 'Britain', 'Ireland', 'California', 'Galapagos', 'Galapagos Islands']
    others = set([
            # old stuff,
            'Capcom', 'New York Times', 'NFL', 'NASA', 'JPL', 'Malin Space Science Systems',
            'SpaceX', 'White House', 'Sony', 'Disney', 'Walt Disney', 'Touchstone', 'Oscars',
            'Sundance', 'Variety', 'Atlantic', 'Instagram', 'Facebook', 'Twitter',
            'West Bromwich Albion', 'Manchester City', 'Crystal Palace', 'Sevilla', 'Östersund', 'Arsenal',
             'Liverpool', 'Norwich City', 'Red Star Belgrade', 'St Johnstone', 'Patriots', 'Seahawks',
             'Oakland Raiders', 'Dallas Cowboys', 'New York Giants', 'Eagles',
            'Seattle', 'Philadelphia', 'New Orleans', 'Minnesota', 'San Francisco', 'Cleveland', 'Arizona',
            'Denver', 'Pyeongchang', 'Seoul', 'Tokyo', 'Cannes', 'Festival', 'New York',
            'Washington DC', 'America', 'Britain', 'Ireland', 'California', 'Galapagos', 'Galapagos Islands',
            # Month, day, date
            'January' , 'Jan', 'February', 'Feb', 'March', 'October', 'November', 'December',
            'Tuesday', 'Wednesday', 'Thursday', 'Tue', 'Friday', 'Wed', 'Today', 'August', 'Monday',
            'Saturday', 'Sunday',  'Week', 'Year',
            # Countries, Locations, Organization, etc
            'Greek', 'Galapagos Islands', 'Capitol Hill', 'American', 'Britain', 'West Antarctic', 
            'British', 'Irish', 'American', 'Chinese', 'Japanese', 'Korean', 'Atlanta', 'America',
            'White House', 'Americans', 'Theatre', 'New York', 'Film Festival', 'NASA', 'Amazon', 'Senate',
            'Chicago', 'London', 'U.S.', 'Vietnam', 'Los Angeles', 'Oslo', 'Academy Award', 'Super Bowl',
            'US', 'Sydney', 'Japan', 'Western', 'Book', 'Paramount', 'Washington Post', 'NRA', 'French',
            'African', 'Cannes Film Festival', 'South', 'East', 'Asian', 'Texas', 'Ukraine', 'Congress',
            'Paris', 'Italian', 'Hungarian', 'Alabama', 'African-American', 'Berlin', 'European',
            'Pentagon', 'English', 'Iraq', 'Florida', 'North Korea', 'CDC', 'East Coast', 'Missouri', 
            'Sacramento', 'Manchester', 'United States', 'Russians', 'The Atlantic', 
            # General words
            'Actress', 'Women', 'Director', 'President', 'Sen.', 'Justice', 'Even', 'First', 'Good',
            'Discovery', 'Get Out', 'Perhaps', 'Original', 'Up', 'Supporting',
            'Instead', 'Before', 'Democratic', 'World', 'Post', 'National', 'Which', 'Maybe',
            'Name', 'While', 'Golden', 'Earth', 'Out', 'Theater', 'Picture', 'United', 'Six', 'Empire',
            'Cinematography', 'Best Original', 'Original Screenplay', 'Studios', 'Area', 'Last', 'Far',
            'Another', 'Day', 'Awakends', 'SEE', 'SEE ALSO', 'Best Supporting', 'Where', 'Once', 'Live',
            'Democrats', 'Nation', 'Entertainment', 'Boys', 'Actor', 'Screenplay', 'Reporter',
            'Super', 'Republicans', 'Given', 'Best Director', 'Room', 'Image', 'Brexit', 'Rep', 'Episode', 
            'Supporting Actress', 'Fahrenheit', 'Christmas', 'PlayStation', 'Visual', 'Express', 'Stay',
            'Meets', 'General', 'Still', 'Episode', 'City', 'Best Picture', 'League', 'World War',
            'During', 'Best Supporting Actress', 'Many', 'Hour', 'Club', 'Under', 'Darkest',
            'Saturday Night Live', 'Excited', 'Best Actor', 'Scene', 'Voices', 'Leave', 'Whether',
            'Republican', 'International', 'Very', 'Go', 'Center', 'Want', 'Opening', 'Wonder', 'Beyond',
            'Design', 'Eye', 'News', 'Leader', 'Dreamers', 'Call', 'Hotel', 'Something', 'Moon',
            'Pride', 'According', 'Both', 'Network', 'Next', 'Stage', 'Well', 'R', 'Everything', 'Seconds',
            # Others
            'Women Project', 'Constitution', 'EPA', 'Environmental Protection Agency', 'Billboards', 'Academy',
            'Outside', 'Mr.', 'Hollywood', 'EST', 'Wars', 'Ms.', 'Star Wars', 'Now', 'Last',
            'Here', 'do', 'Panther', "I'm", 'New', 'The Last', 'Award', 'War', 'Water', 'Awards',
            'Monster', 'Film', 'Trailer', 'WATCH', 'Netflix', 'House', 'Though', 'One', 'Times',
            'York', 'Oscars', 'Oscar',  'Over', 'Lady', 'Time', 'Three', 'ALSO', 'Yes', 
            'Best Pictures', 'Game', 'TV', 
            # Special,
            "'", "'Black", "I'd", "'Black Panther'", 'Jedi', 'Your Name', 'Last Jedi', 'The Last Jedi',
            "Panther'", "I've", 'Outside Ebbing', 'Justice League', 'Monster Hunter', "We're", "A.",
            'Darkest Hour', 'The Post', 'Call Me', 'MoviePass', 'Panther Week', 'Three Billboards Outside', 
            'The Force Awakens', "I'll", 'Three Billboards', 'Three Billboards Outside Ebbing', 'II',
            'The New York', 'X', 'York Times', "We've", 'Night Live', 'Black Panther Week', "You'll"
            ])
                  
    for word in organizations:
        if word.lower() == word_string.lower():
            return 1

    for word in clubs:
        if word.lower() == word_string.lower():
            return 1

    for word in locations:
        if word.lower() == word_string.lower():
            return 1
    
    if word_string in others:
        # print('in others = ' + word_string)
        return 1

    return 0

def isNearParentheses(words, start, end, word_string):
    # check if the candidate name is next to a ( or )
    keys = ['(', ')']
    flag = 0

    if not isCapitalized(word_string):
        return flag

    if start > 0:
        preceding_str = words[start - 1]
        if preceding_str in keys:
            flag = 1
    if end < len(words):
        succeeding_str = words[end]
        if succeeding_str in keys:
            flag = 1

    return flag

def inNamelist(words, start, end, word_string):
   # check if candidate name is in a list of names in the documents
   # eg: Trang, Hoai, and Isaac, or Trang/Hoai/Isaac
   keys = [',', 'and', 'or', '/']
   flag = 0
   if not isCapitalized(word_string):
       flag = 0
   elif start > 0 and end < len(words):
       preceeding_str = words[start - 1]
       succeeding_str = words[end]
       if preceeding_str in keys:
           if start > 1 and isCapitalized(words[start - 2]):
               flag = 1
       if succeeding_str in keys:
           if end < len(words) - 1 and isCapitalized(words[end + 1]):
               flag = 1
   return flag


def get_word_frequency(words, start_tag, end_tag):
    # return a dictionary of word in words
    untagged_words = []
    for word  in words:
        untagged_word = removeTags(word, start_tag, end_tag)
        untagged_words.append(untagged_word)
    word_frequency = collections.Counter(untagged_words)
    return word_frequency, untagged_words


def contains_legislator_name(word_string):
    for word in word_string.split():
        if word in LEGISLATORS_NAMES_DICT:
            return 2

    return 0


def generate_legislator_names():
    names = {}
    with open(LEGISLATORS_CURRENT_CSV, 'r') as file:
        rd = csv.reader(file, delimiter=",", quotechar='"')
        for row in rd:
            if len(row) > 0:
                names[row[0]] = 1
                names[row[1]] = 1

    with open(LEGISLATORS_HISTORICAL_CSV, 'r') as file:
        rd = csv.reader(file, delimiter=",", quotechar='"')
        for row in rd:
            if len(row) > 0:
                names[row[0]] = 1
                names[row[1]] = 1

    return names


def contains_actor_name(word_string):
    for word in word_string.split():
        if word in ACTOR_NAMES_DICT:
            return 2

    return 0


def generate_actor_names():
    names = {}
    with open(ACTOR_NAMES1_CSV, 'r') as file:
        for line in file:
            name_parts = line.split()
            for name in name_parts:
                names[name] = 1

    with open(ACTOR_NAMES2_CSV, 'r') as file:
        for line in file:
            name_parts = line.split()
            for name in name_parts:
                names[name] = 1
    return names


def contains_first_name(word_string):
    for word in word_string.split():
        if word in FIRST_NAMES_DICT:
            return 2

    return 0


def generate_first_names():
    names = {}
    with open(FIRST_NAMES_CSV, 'r') as file:
        for line in file:
            words = line.split()
            if len(words) > 0:
                names[capitalize_as_name(words[0])] = 1
    return names


def capitalize_as_name(name):
    capitalized_name = ""

    for i in range(0, len(name)):
        if i == 0:
            capitalized_name += name[i]
        else:
            capitalized_name += name[i].lower()

    return capitalized_name


def is_title(word):
    if word in TITLES_DICT:
        return 1
    else:
        return 0


def generate_titles():
    titles = {}
    with open(TITLES_CSV, encoding = 'utf8') as file:
        for line in file:
            titles[line.strip()] = 1
    return titles


def is_name_suffix(word):
    suffixes = ['Sr', 'Sr.', 'Jr', 'Jr.', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                'sr', 'sr.', 'jr', 'jr.', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']
    words = word.split()
    if len(words) > 0 and words[-1] in suffixes:
        return 1
    else:
        return 0


def is_common_word(word):
    stopwords = ['\'s', 'I', 'A', 'An', 'The', 'But', 'If', 'So', 'He', 'She', 'They', 'There', 'Are', 'Is', 'Be',
                 'You', 'Able', 'About', 'Across', 'After', 'All', 'Almost', 'Also', 'Am', 'Among', 'And', 'Any', 'As',
                 'At', 'Because', 'Been', 'By', 'Best', 'Can', 'Cannot', 'Could', 'Dear', 'Did', 'Do', 'Does', 'Either',
                 'Else', 'Ever', 'Every', 'For', 'From', 'Get', 'Got', 'Had', 'Has', 'Have', 'He', 'Her', 'Hers', 'Him',
                 'His', 'How', 'However', 'In', 'Into', 'It', 'Its', 'Just', 'Least', 'Let', 'Like', 'Likely', 'May',
                 'Me', 'Might', 'Most', 'Must', 'My', 'Neither', 'No', 'Nor', 'Not', 'Of', 'Off', 'Often', 'On', 'Only',
                 'Or', 'Other', 'Our', 'Own', 'Rather', 'Said', 'Say', 'Says', 'She', 'Should', 'Since', 'Some', 'Than',
                 'That', 'Their', 'Then', 'These', 'This', 'To', 'Too', 'Us', 'Wants', 'Was', 'Who', 'Whom', 'We', 'When',
                 'Why', 'What', 'With', 'Would', 'Yet', 'Your']
    if word in stopwords:
        return 1
    else:
        return 0


def contains_common_word(word_string):
    for word in word_string.split():
        if is_common_word(word) == 1:
            return 1

    return 0


def is_near_capitalized(words, start, end):
    '''
    Checks to see if the words are near another capitalized word (that could be part of the name) but not counting
    common prefixes like Dr., Sen., etc
    '''
    near_capitalized = 0
    if start > 0:
        if len(words[start-1]) > 0 and words[start-1][0].isupper() and not checkPrefix(words[start-1]):
            near_capitalized = 1

    if end < len(words):
        if len(words[end]) > 0 and words[end][0].isupper():
            near_capitalized = 1

    return near_capitalized


def checkTitlePrefix(word):
    title_words = ['Dr.', 'Esq.', 'Hon.', 'Jr.', 'Mr.', 'Mrs.', 'Ms.', 'Messrs.',
                   'Mmes.', 'Msgr.', 'Prof.', 'Rev.', 'Rt. Hon.', 'Sr.', 'St.', 'Sen.', 'Sens.']
    # check if word is a title prefix
    if word in title_words:
        return True
    else:
        return False


def checkHyphen(word):
    # return true if string has punctuations in cases like Billings-Lads
    character_frequency = collections.Counter(word)
    if word[0].isalpha() and word[-1].isalpha() and character_frequency['-'] == 1 and containUppercase(word):
        return True
    else:
        return False


def checkApostrophe(word):
    # return true if string is name with apostrophe lik O'Dowd
    character_frequency = collections.Counter(word)
    if character_frequency['\''] == 1 and containUppercase(word):
        return True
    else:
        return False


def contains_punctuation(word):
    retval = False
    for char in word:
        if char in string.punctuation:
            retval = True
            break
    return retval


def contains_punctuation_except_some(word):
    '''
    Returns true if the word contains a punctuation besides . or ' or - (a name might contain a . or ' or -, e.g. O'Dowd or J.F. Billings-Ladson)
    '''
    retval = False
    words = word.split()  # split string by space
    punctuation = [c for c in string.punctuation if c != '.' and c != '\'' and c != '-']
    if is_name_suffix(word):
        retval = False  # do not get rid of string with name suffix
    else:
        for word in words:
            if checkTitlePrefix(word):
                retval = True  # get rid of this
                break
            elif contains_punctuation(word):
                if checkHyphen(word) or checkApostrophe(word):
                    retval = False
                    break
                else:
                    for char in word:
                        if char in punctuation:
                            retval = True
                            break
    return retval


def num_of_labels(data):
    '''
    Returns the number of positive labels in the data
    :param data: the list of data instances
    :return: an int value of the number of positive labels in the data
    '''
    sum = 0
    for instance in data:
        sum += instance[-1]
    return sum


def split_string(word_string):
    '''
    Split string into words and punctuations except for string with tags and
    special words. Return a list of string
    '''
    start_tag = '<pname>'
    end_tag = '</pname>'
    title_words = ['Dr.', 'Esq.', 'Hon.', 'Jr.', 'Mr.', 'Mrs.', 'Ms.', 'Messrs.',
                   'Mmes.', 'Msgr.', 'Prof.', 'Rev.', 'Rt. Hon.', 'Sr.', 'St.', 'Sen.', 'Sens.']
    specials = ['U.S.'] # can extend this list
    start_tag_id = word_string.find(start_tag)
    end_tag_id = word_string.find(end_tag)
    split_words = []
    word_string_frequency = collections.Counter(word_string)
    # if word_string only has 1 character, return it
    if len(word_string) < 2:
        split_words.append(word_string)
        return split_words
    # if word_string is all alphabets, or all digits, or can be convert to a float, or alphanumeric, return it
    elif word_string.isalpha() or word_string.isdigit() or word_string.isalnum() or isNumber(word_string):
        split_words.append(word_string)
        return split_words
        # if word_string starts and ends with alphabetic character but has a hyphen in the middle
    elif word_string[0].isalpha() and word_string[-1].isalpha() and word_string_frequency['-'] == 1 and containUppercase(word_string):
        split_words.append(word_string)
        return split_words
    # if word_string starts or ends with tags, return it
    elif start_tag_id == 0:
        if word_string[end_tag_id:] == end_tag:
            split_words.append(word_string)
            return split_words
        elif end_tag_id > 0 and word_string[end_tag_id:] != end_tag:
            split_pos = end_tag_id + len(end_tag)
            word1 = word_string[:split_pos]
            word2 = word_string[split_pos:]
            split_word2 = split_string(word2)
            split_word2.insert(0, word1)
            return split_word2
        elif end_tag_id == -1:
            split_words.append(word_string)
            return split_words
    # if word_string has start_tag but it does not start with start_tag, split it
    elif start_tag_id > 0:
        word1 = word_string[0:start_tag_id]
        word2 = word_string[start_tag_id:]
        split_word1 = split_string(word1)
        split_word2 = split_string(word2)
        split_word1.extend(split_word2)
        return split_word1
    elif start_tag_id == -1:
        if word_string[end_tag_id:] == end_tag:
            split_words.append(word_string)
            return split_words
        elif end_tag_id > 0 and word_string[end_tag_id:] != end_tag:
            split_pos = end_tag_id + len(end_tag)
            word1 = word_string[:split_pos]
            word2 = word_string[split_pos:]
            split_word2 = split_string(word2)
            split_word2.insert(0, word1)
            return split_word2
        elif end_tag_id == -1:
            # if word_string is not alphanumeric, split it
            # find first position that is not alphanumeric
            first_nonalnum = find_first_nonalnum(word_string)
            # if first_nonalnum character is a period, make sure it's not part of title words or initials
            if word_string[first_nonalnum] == '.':
                if word_string in title_words:
                    split_words.insert(0, word_string)
                    return split_words
                elif len(word_string) == 2 and word_string[0].isupper():
                    split_words.insert(0, word_string)
                    return split_words
                elif word_string in specials:
                    split_words.insert(0, word_string)
                    return split_words
                else:
                    word1 = word_string[0:first_nonalnum]
                    word2 = word_string[first_nonalnum]
                    word3 = word_string[first_nonalnum+1:]
                    if len(word1) > 0:
                        split_word1 = split_string(word1)
                    else:
                        split_word1 = word1
                    if len(word3) > 0:
                        split_word3 = split_string(word3)
                    else:
                        split_word3 = word3
                    if len(split_word1) > 0:
                        split_word1.append(word2)
                        if len(split_word3) > 0:
                            split_word1.extend(split_word3)
                        return split_word1
                    else:
                        if len(split_word3) > 0:
                            split_word3.insert(0, word2)    
                            return split_word3
            # if first_nonalnum character is an apostrophe, make sure it's not part of the 's
            elif word_string[first_nonalnum] == '\'':
                if len(word_string) == 2 and word_string[-1] == 's':
                    split_words.append(word_string)
                    return split_words
                elif len(word_string) > 2 and word_string[-1] == 's':
                    word1 = word_string[0:first_nonalnum]
                    word2 = word_string[first_nonalnum:]
                    split_words.append(word1)
                    split_words.append(word2)
                    return split_words
                else:
                    split_words.append(word_string)
                    return split_words
            else:            
                word1 = word_string[0:first_nonalnum]
                word2 = word_string[first_nonalnum]
                word3 = word_string[first_nonalnum+1:]
                if len(word1) > 0:
                    split_word1 = split_string(word1)
                else:
                    split_word1 = word1
                if len(word3) > 0:
                    split_word3 = split_string(word3)
                else:
                    split_word3 = word3
                if len(split_word1) > 0:
                    split_word1.append(word2)
                    if len(split_word3) > 0:
                        split_word1.extend(split_word3)
                    return split_word1
                else:
                    if len(split_word3) > 0:
                        split_word3.insert(0, word2)    
                        return split_word3

def find_first_nonalnum(word):
    for char in word:
        if not char.isalnum():
            return(word.index(char))

def isNumber(word):
    try:
        float(word)
        return True
    except ValueError:
        return False

def containUppercase(word):
    retval = False
    for char in word:
        if char.isupper():
            retval = True
            break
    return retval

def isCapitalized(word_string):
    #print (word_string)
    # nobiliary particles that aren't always capitalized (from Wikipedia: nobiliary particles)
    particles = ['de', 'del', 'van', 'von', 'af', 'du', 'd', 'des', 'zu',
                 'do', 'dos', 'da', 'das', 'di', 'der', 'bin', 'al', 'la']
    
    word_string = word_string.split()
    flag = 1 # capitalized
    # some names like Neil deGrasse Tyson might be thrown out by this rule
    # if only check first letter
    # check if each word contains at least one capitalized letter
    for word in word_string:
        if not containUppercase(word) and not (word in particles or (word[0] == '\'' and word != '\'s')):
            flag = 0 # not capitalized
            break
    return flag
            
def removeTags(word_string, start_tag, end_tag):
    # remove markedup tags
    if start_tag in word_string:
        word_string = word_string.replace(start_tag, '')
    if end_tag in word_string:
        word_string = word_string.replace(end_tag, '')
    return word_string

def checkPrefix(word):
    # check if word_string has a title
    title_prefixes = ['Dr.', 'Esq.', 'Hon.', 'Jr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Rev.',
                'Sr.', 'St.', 'Dr', 'Esq', 'Hon', 'Jr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Rev',
                'Sr', 'St', 'Sen.', 'Sen', 'Sens.', 'Sens', 'Lady', 'Lord', 'Captain', 'President',
                'General', 'Doctor', 'Professor', 'Senator', 'Senators',
                'Father', 'Reverend', 'Earl', 'Mister', 'Miss', 'Madam', 'Chancellor',
                'Vice-President', 'Dean', 'Pope', 'Rabbi', 'Prince', 'Queen', 'Princess', 'Leader',
                'Whip', 'Bishop', 'Chairman', 'CEO', 'DJ', 'GM', 'Democrat', 'Republican', 'Democratic',
                'Independent', 'Candidate']
    occupational_prefixes = ['representative', 'congressman', 'congresswoman',
                'director', 'composer', 'actor', 'actress', 'chief', 'detective', 'screenwriter',
                'producer', 'writer', 'by', 'screenwriter', 'investigator',
                'couturier', 'musician', 'artist', 'photographer', 'keyboardist',
                'bassist', 'drummer', 'singer', 'biologist', 'guitarist',
                'surgeon', 'secretary', 'curator', 'archivist', 'diva', 'publicist',
                'painter', 'designer', 'reporter', 'prodigy', 'medalist',
                'pilot', 'brakewoman', 'manager', 'goalkeeper']
    family_prefixes = ['son', 'daughter', 'father', 'mother', 'grandmother', 'cousin', 'grandfather',
                       'nephew', 'niece', 'sister', 'brother', 'uncle', 'aunt', 'law']
    verbs = ['said', 'wrote', 'explained', 'added', 'admitted', 'claimed']
    
    if word in title_prefixes or word.lower() in occupational_prefixes or word.lower() in verbs:
        return 1
    else:
        return 0

def checkSuffix(word):
    # check if word_string followed by
    suffixes = ['says', 'said', 'talk', 'talked', 'writes', 'wrote', 'plays', 'played',
                'believes', 'believed', 'explains', 'explained', '\'s', 'who', 'starred',
                'whose', 'the', 'added', 'claims', 'claimed', 'Esq.']
    
    word = word.lower()
    if word in suffixes:
        return 1
    else:
        return 0
        
def checkOthers(words, before_index, after_index):
    # check if word_string maybe a location name, movie name, or an organization
    preceeding_words = ['the', 'in', 'on', 'at']
    succeeding_words = ['avenue', 'city', 'street', 'st', 'ave', 'town', 'village']
    other_words = ['theater', 'theaters', 'theatre', 'theatres', 'studio', 'studios', 'hotel', 'bank', 'cinema', 'cinemas',
                   'west', 'east', 'north', 'south', 'western', 'eastern', 'northern', 'southern', 'conference', 'festival']
    
    if words[before_index].lower() in preceeding_words or words[after_index].lower() in succeeding_words:
        return 1
    else:
        for word in words[before_index+1:after_index]:
            if word.lower() in other_words:
                return 1
        return 0

def findClassLabel(word_string, start_tag, end_tag):
    # find the class label for the word_string
    class_label = 0 # negative label
    start_tag_id = word_string.find(start_tag)
    end_tag_id = word_string.find(end_tag)
    if start_tag_id == 0 and end_tag_id > 0 and word_string[end_tag_id:] == end_tag:
        class_label = 1 # positive label
    return class_label

def createDevAndTestFileSet():
    """ Shuffles the marked-up file set and divide it into two for training and testing.
    Returns 'train_file_names' and 'test_file_names'."""
    train_file_names = []
    test_file_names = []

    #file_names = []

    '''
    for file_name in os.listdir(MarkedUp):
        if file_name != '.DS_Store':
            file_names.append(file_name)
    '''

    for file_name in os.listdir(TrainingSet):
        train_file_names.append(file_name)
    for file_name in os.listdir(TestSet):
        test_file_names.append(file_name)
    '''
     shuffle the list, and create training and testing list
    random.seed(1) # fix the seed for debugging purposes
    random.shuffle(file_names)
    return file_names[0:int(len(file_names) * 0.66)], file_names[int(len(file_names) * 0.66):]
    '''
    return train_file_names, test_file_names

'''
def writeToCSV(filename):
    global LOCATION_FEATURES
    global OTHER_FEATURES
    headers = ['string_id', 'string'] + LOCATION_FEATURES + OTHER_FEATURES + ['class_label']

    if os.path.isfile(MarkedUp + filename):
        text = clean_file(filename)
        data = data_generator(filename, text)
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(csv_file, encoding='utf-8', header=True, index=False)
'''

def copyFile(fromPath, toPath):
    '''
    Copies the file from fromPath to toPath
    '''
    with open(fromPath, 'r') as inFile:
        with open(toPath, 'w') as outFile:
            data = inFile.read()
            outFile.write(data)

def extractAndCreateCSV(file_names, csv_file, directory):
    """Scan all the files in file_names and produces a single CSV file that
    containing strings, feature vectors, and class_label."""
    global LOCATION_FEATURES
    global OTHER_FEATURES
    headers = ['string_id', 'string', 'filename'] + LOCATION_FEATURES + OTHER_FEATURES + ['class_label']
    print('creating csv file:' + csv_file)
    # open MarkedUp folder and process all files
    fileID = 0
    for filename in file_names:
        #print ("processing file.. ", filename)
        if filename == '.DS_Store':
            continue
        if os.path.isfile(MarkedUp + filename):
            #copyFile(MarkedUp + filename, directory + filename)
            text = clean_file(filename)
            data = data_generator(fileID, filename, text)
            df = pd.DataFrame(data, columns = headers)
            fileID = fileID + 1
            # if csv_file is already exist, open it and append new data
            if os.path.isfile(csv_file):
                # check if filename is already processed:
                existing_df = pd.read_csv(csv_file)
                file_list = list(existing_df['filename'])
                if filename in file_list:
                    print ('Skip file: ', filename)
                    continue
                else:                
                    df.to_csv(csv_file, mode = 'a', encoding = 'utf-8', header = False, index = False)
            else:
                df.to_csv(csv_file, encoding = 'utf-8', header = True, index = False)

def cleanMarkedUpFiles():
    for filename in os.listdir(MarkedUp):
        clean_file(filename)

def main():
    global TEST_CSV
    global TRAIN_CSV
    global TITLES_DICT
    global FIRST_NAMES_DICT
    global ACTOR_NAMES_DICT
    global LEGISLATORS_NAMES_DICT
    TITLES_DICT = generate_titles()
    FIRST_NAMES_DICT = generate_first_names()
    ACTOR_NAMES_DICT = generate_actor_names()
    LEGISLATORS_NAMES_DICT = generate_legislator_names()
    train_input_files, test_input_files = createDevAndTestFileSet()
    #print(train_input_files)
    #print(test_input_files)    
    extractAndCreateCSV(train_input_files, TRAIN_CSV, DATA + 'trainingSet/')
    extractAndCreateCSV(test_input_files, TEST_CSV, DATA + 'testSet/')
    #extractAndCreateCSV(['stage1_docs/Data/Cleaned_MarkedUp/cleaned_hl_105_m.txt',
    #                     '/tmp/result.csv', "");
    #extractAndCreateCSV(['test.txt'], '/tmp/result.csv', '')

                
if __name__ == "__main__":
    main()
