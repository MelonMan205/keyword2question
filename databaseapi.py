import json
import os
from main import semantic_searcher
from main import verbose
# import asyncio
# import aiohttp

with open('vcaa_exams_major.json', 'r') as f:
    data = json.load(f)


import re
def getURL(subject, year_range):
    urls = []

    subject = semantic_searcher.find_best_subject_match(subject, tuple(data.keys()))
    if subject:
        for year in year_range:
            if year in data[subject] and subject in data:
                urls.append(data[subject][year])
                
                #urls.append(data[re.search((rf'({subject})'), string=(i for i in data.keys()))][year])
            else:
                
                print(f"Error: URL not found for {subject} in {year}") if verbose else None
    else:
        print(f"Error: No matching subject found for '{subject}'") if verbose else None      
    return urls, year_range, subject

def generateRange(arr: list):
    return [str(i) for i in range(arr[0], arr[1]+1, 1)]



# subject = 'Physics'
# year = '2023'
# print((data[subject]) if year in data[subject] and subject in data else "Error: URL not found")
