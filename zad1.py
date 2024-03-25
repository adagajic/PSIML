import os
import re
from datetime import datetime

def check_for_logtxt_in_dir(dir):
    # get all files in directory
    files = os.listdir(dir)
    # count the number of .logtxt files. Do it recursively if there are directories in the directory
    count = 0
    for file in files:
        if os.path.isdir(os.path.join(dir, file)):
            count += check_for_logtxt_in_dir(os.path.join(dir, file))
        elif file.endswith('.logtxt'):
            count += 1
    return count
def get_all_logtxt_files(dir):
    # get all files in directory
    files = os.listdir(dir)
    logtxt_files = []
    # get all .logtxt files
    for file in files:
        if os.path.isdir(os.path.join(dir, file)):
            logtxt_files += get_all_logtxt_files(os.path.join(dir, file))
        elif file.endswith('.logtxt'):
            logtxt_files.append(os.path.join(dir, file))
    return logtxt_files
# create main

keywords_for_error = ['<err>', '[error]', 'ERROR', 'loglevel=error', 'fatal-error']
files_with_error = []
# get input as string
dir = input()
# count the number of .logtxt files. Do it recursively if there are directories in the directory
logtxt_files = get_all_logtxt_files(dir)
# print the result in one line
count = len(logtxt_files)
print(count)
# calculate the number of all log entries(non empty lines in logtxt files)
log_entries = 0
full_messages = []
bll = True
for file in logtxt_files:
    with open(file) as f:
        for line in f:
            if line.strip():
                log_entries += 1
                full_messages.append(line)
            if bll and any(keyword in line for keyword in keywords_for_error):
                files_with_error.append(file)
                bll = False
    bll = True
        
print(log_entries)

# get all files with error
print(len(files_with_error))

pattern1 = r'<([a-z]+)>(.*)'
pattern2 = r' --- (.*)'
pattern3 = r' - (.*)'
pattern4 = r'msg=(.*)'

# get all msg from logtxt files using patterns above

messages = []
for line in full_messages:
    match1 = re.search(pattern1, line)
    match2 = re.search(pattern2, line)
    match3 = re.search(pattern3, line)
    match4 = re.search(pattern4, line)
    if match1:
        messages.append(match1.group(2))
    elif match2:
        messages.append(match2.group(1))
    elif match3:
        messages.append(match3.group(1))
    elif match4:
        messages.append(match4.group(1))

# get 5 most common words in messages
dict = {}
for msg in messages:
    # replace ', ' with ' '
    msg = msg.replace(', ', ' ')
    msg = msg.replace('. ', ' ')
    msg = msg.replace(',', ' ')
    msg = msg.replace('.', ' ')
    msg = msg.replace(';', ' ')
    msg = msg.replace(':', ' ')
    msg = msg.replace('-', ' ')
    # split the message by space and count the number of each 
    # word in the message and add to the dictionary
    words = msg.split()
    # remove repeated words
    words = list(set(words))
    for word in words:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1
# get 5 most common words, if same number of occurences, sort them alphabetically
most_common_words = sorted(sorted(dict, key=lambda x: x), key=lambda x: dict[x], reverse=True)[:5]
print(', '.join(map(str, most_common_words)))
# covert date to seconds
# format of date can be:
# 2024 02 25 05:04:54
# dt=2024-02-25_05:04:37
# 25.02.2024.05h:22m:24s
# 25.02.2024.06:06:41
# [2024-02-25 07:12:38]
date_type_1 = r'(\d{4}) (\d{2}) (\d{2}) (\d{2}):(\d{2}):(\d{2})'
data_type_2 = r'dt=(\d{4})-(\d{2})-(\d{2})_(\d{2}):(\d{2}):(\d{2})'
date_type_3 = r'(\d{2}).(\d{2}).(\d{4}).(\d{2})h:(\d{2})m:(\d{2})s'
date_type_4 = r'(\d{2}).(\d{2}).(\d{4}).(\d{2}):(\d{2}):(\d{2})'
data_type_5 = r'\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\]'

msg_by_date = []
for msg in full_messages:
    
    # get the date from the message
    
    match1 = re.search(date_type_1, msg)
    match2 = re.search(data_type_2, msg)
    match3 = re.search(date_type_3, msg)
    match4 = re.search(date_type_4, msg)
    match5 = re.search(data_type_5, msg)
    if match1:
        year, month, day, hour, minute, second = match1.groups()
    elif match2:
        year, month, day, hour, minute, second = match2.groups()
    elif match3:
        day, month, year, hour, minute, second = match3.groups()
    elif match4:
        day, month, year, hour, minute, second = match4.groups()
    elif match5:
        year, month, day, hour, minute, second = match5.groups()
    else:
        print(msg)
    # convert date to seconds
    date = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    msg_by_date.append((date, msg))
    
msg_by_date.sort(key=lambda msg: msg[0])
warning_keywords = ['<warn>', '[warning]', 'WARN', 'loglevel=warning', 'the-warning']
number_of_warnings = 0
count_first = 0
count_last = 0
longest_time = 0
# for msg in msg_by_date:
#     print(msg)
for msg in msg_by_date:
    if any(keyword in msg[1] for keyword in warning_keywords):
        number_of_warnings += 1
        break
    count_first += 1
count_last = count_first+1
while(count_last < len(msg_by_date) and number_of_warnings < 5):
    if any(keyword in msg_by_date[count_last][1] for keyword in warning_keywords):
        number_of_warnings += 1
        longest_time = max(longest_time, (msg_by_date[count_last][0] - msg_by_date[count_first][0]).total_seconds())
    if number_of_warnings != 5:
        count_last += 1
count_first += 1
count_last += 1
while(count_last < len(msg_by_date)):
    while(count_last < len(msg_by_date)):
        if any(keyword in msg_by_date[count_last][1] for keyword in warning_keywords):
            break     
        count_last += 1
    while count_first < count_last:
        if any(keyword in msg_by_date[count_first][1] for keyword in warning_keywords):
            break
        count_first += 1
    if count_last < len(msg_by_date):
        longest_time = max(longest_time, (msg_by_date[count_last][0] - msg_by_date[count_first][0]).total_seconds())
    count_first += 1
    count_last += 1
    
    

print(int(longest_time))    