# Collecting results from saved search from CrowdTangle

import os
import pandas as pd
from pytangle.api import API

# Token
token = 'ACCSESS TOKEN'
api = API(token=token)

# Print list of saved search
for a_list in api.lists():
    print(a_list)

# Ids of the saved searches
list_ids = [
    {'id': 1234467, 'title': 'climate', 'type': 'SAVED_SEARCH'}
]

# Set up date range
date_range = pd.date_range(start='1/1/2019', end='2/1/2021', freq='MS')

# Output directory
out_dir = r'output folder to collect CrowdTangle data'

for id_dict in list_ids:
    search_id = id_dict['id']
    search_rule = id_dict['title']
    for start, end in zip(date_range, date_range[1:]):
        print(start.date().isoformat(), end.date().isoformat(), search_rule)
        start_date = start.date().isoformat()
        end_date = end.date().isoformat()
        ym = start_date[:7]

        # Fetch data by saved search and time range
        data = []
        for n, a_post in enumerate(api.posts(listIds=[search_id],
                                             # Set to -1 to fetch everything
                                             count=-1,
                                             batchSize=100,
                                             # Set sortBy to date to fetch everything in time range
                                             sortBy='date',
                                             startDate=start_date,
                                             endDate=end_date,
                                             timeframe=None,
                                             # Language restriction
                                             language='en',
                                             )):
            data.append(a_post)
        # Flatten data
        df = pd.json_normalize(data)

        # Name & ship out
        out_name = search_rule + '_' + start_date + '_' + end_date + '.csv'
        df.to_csv(os.path.join(out_dir, out_name))
