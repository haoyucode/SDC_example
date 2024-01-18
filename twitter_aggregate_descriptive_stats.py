# Aggregate monthly Twitter descriptive statistics due to the large size.

import pandas as pd
import os
import glob
import re
import datetime
import addfips

# Updated on 02/13/2023
# Aggregate monthly descriptive statistics
# Add monthly VADER, GeoLocated Tweets, and count by State

if __name__ == '__main__':
    author = 'SDC'

    desc_dir = r'path to monthly descriptive statistics'
    desc_fn = [f for f in glob.glob(os.path.join(desc_dir, '*.xlsx')) if re.search(r'\d{6}', f)]
    desc_fn = sorted(desc_fn)

    out_path = r'output path of aggregated descriptive statistics_{}-{}.xlsx'.format(
        desc_fn[0].split('\\')[-1].split('.')[0],
        desc_fn[-1].split('\\')[-1].split('.')[0]
    )

    stat_keys = ['volume_monthly', 'volume_daily', 'vader_monthly', 'vader_daily',
                 'geolocated_monthly', 'count_by_state',
                 'hashtag', 'retweet', 'user', 'mention', 'url',
                 'emoji', 'unigram', 'bigram', 'trigram']

    stat_dfs = {k: pd.DataFrame() for k in stat_keys}

    # Documentation
    analytic_file_path = os.path.dirname(
        pd.read_excel(desc_fn[0], sheet_name='documentation', header=None).iloc[0,1]
    )

    documentation = pd.DataFrame({
        'names': ['analytic_path', 'monthly_descriptive_path', 'author', 'date_created'],
        'values': [analytic_file_path, desc_dir, author, datetime.date.today()]
    })

    for fn in desc_fn:
        print(datetime.datetime.now(), ' Reading ', fn)
        for sheet in stat_keys:
            if sheet in ['volume_daily', 'vader_daily']:
                df = pd.read_excel(fn, sheet_name=sheet)
                df['date'] = pd.to_datetime(df['date'])
                df['date'] = df['date'].dt.date
            else:
                df = pd.read_excel(fn, sheet_name=sheet)
            stat_dfs[sheet] = pd.concat([stat_dfs[sheet], df])

    sheet_map = {'unigram': '1-gram', 'bigram': '2-gram', 'trigram': '3-gram'}
    final_agg = {}

    print(datetime.datetime.now(), ' Aggregating')
    for sheet in stat_keys[5:]:
        agg_name = sheet_map.get(sheet, sheet)
        if sheet == 'retweet':
            final_agg['retweet'] = stat_dfs['retweet'].groupby(agg_name).agg({'datetimecreated': 'first', 'count': 'sum'})
        elif sheet == 'emoji':
            final_agg['emoji'] = stat_dfs['emoji'].groupby(['emoji_visualized', 'emoji_name', 'emoji_unicode']).agg({'count': 'sum'})
        elif sheet == 'count_by_state':
            final_agg['count_by_state'] = stat_dfs['count_by_state'].groupby('State FIPS').agg({'tweets': 'sum'})
        else:
            final_agg[sheet] = stat_dfs[sheet].groupby(agg_name).agg('sum')

    # Add state abbreviation and name for the ease of reading
    fips = pd.read_csv(addfips.__path__[0]+"/data/states.csv")[['fips', 'postal']]
    fips.columns = ['State FIPS', 'State Abbreviation']
    final_agg['count_by_state'] = pd.merge(fips, final_agg['count_by_state'], how='right', on='State FIPS')
    final_agg['count_by_state'].set_index('State FIPS', inplace=True)

    print(datetime.datetime.now(), ' Writing to ', out_path)
    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    documentation.to_excel(writer, sheet_name='documentation', index=False)
    for sheet in stat_keys[:5]:
        stat_dfs[sheet].to_excel(writer, sheet_name=sheet, index=False)
    for sheet in stat_keys[5:]:
        if sheet == 'user':
            c = 'posts'
        elif sheet == 'count_by_state':
            c = 'tweets'
        else:
            c = 'count'
        out_df = final_agg[sheet].sort_values(c, ascending=False).reset_index().head(10000)
        out_df.to_excel(writer, sheet_name=sheet, index=False)
    writer.save()
    writer.close()
    print('Finished')


