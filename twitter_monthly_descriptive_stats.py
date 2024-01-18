# Calculate Descriptive Statistics for Monthly Twitter data

import pandas as pd
import datetime
import os
from nltk.corpus import stopwords
import glob
from nltk.util import ngrams
from collections import Counter
import nltk
import regex as re
import numpy as np
import emoji as emj

from text_process_util import text_process, add_gram, make_gram_df, domain_pat

## updated on 02/22/2023
# To be more compatible with current monthly aggregation codes
# Add VADER, geolocated tweets, and count by state and county
# Add retweet creation time

def pd_stack_count(df, grp_cols, grp_name, topn=None):
    if topn:
        counts = df[grp_cols].stack().value_counts().reset_index().head(topn)
    else:
        counts = df[grp_cols].stack().value_counts().reset_index()
    counts = counts.rename(columns={'index': grp_name, 0: 'count'})
    return counts

def fix_truncate_rt(df):
    df.loc[df['bodyoriginal'].notnull(), 'bodypost'] = df.loc[
        df['bodyoriginal'].notnull()].apply(
        lambda r: 'RT @' + r['entitiesusrmentionssname01'] + ': ' + r['bodyoriginal'], axis=1)

def tweet_pd_stats(df, topn=10000):
    print(datetime.datetime.now(), 'calculating descriptive statistics')
    total_tweets = len(df)

    hash_cols = [c for c in df.columns if 'entitieshtagstext' in c]
    mention_cols = [c for c in df.columns if 'entitiesusrmentionssname' in c]

    retweet_flag = df['bodyoriginal'].notnull()
    rt_df = df.loc[retweet_flag, ['objpostedTime', 'bodypost']]
    rt_df['datetimecreated'] = pd.to_datetime(rt_df['objpostedTime'], infer_datetime_format=True).dt.tz_localize(None)
    rt_count = rt_df.groupby('bodypost').agg({'datetimecreated': 'first', 'bodypost': 'size'})
    rt_count = rt_count.rename(columns={'bodypost': 'count'})
    rt_count = rt_count.sort_values('count', ascending=False)
    rt_count = rt_count.reset_index()
    rt_count = rt_count.rename(columns={'bodypost': 'retweet'})
    url_count = df['gnipexpandedurl'].value_counts().reset_index()
    url_count = url_count.rename(columns={'index': 'url', 'gnipexpandedurl': 'count'})
    hash_count = pd_stack_count(df[hash_cols].applymap(lambda s: str(s).lower() if pd.notnull(s) else s),
                                hash_cols, 'hashtag')
    mention_count = pd_stack_count(df, mention_cols, 'mention')

    user_count = df['actorpreferredusername'].value_counts().reset_index()
    user_count = user_count.rename(columns={'index': 'user', 'actorpreferredusername': 'posts'})

    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    volume_monthly = df.pivot_table(
        index=pd.Grouper(key='datetime', freq='M'),
        values=['Idpost', 'actorpreferredusername'],
        aggfunc={'Idpost': 'count', 'actorpreferredusername': lambda x: len(x.unique())}).reset_index()
    volume_monthly['datetime'] = yearmo
    volume_monthly = volume_monthly.rename(columns={'datetime': 'yearmo',
                                                    'Idpost': 'posts',
                                                    'actorpreferredusername': 'unique_users'})

    volume_daily = df.pivot_table(
        index=pd.Grouper(key='datetime', freq='D'),
        values=['Idpost', 'actorpreferredusername'],
        aggfunc={'Idpost': 'count', 'actorpreferredusername': lambda x: len(x.unique())}).reset_index()
    volume_daily['datetime'] = volume_daily['datetime'].dt.date
    volume_daily = volume_daily.rename(columns={'datetime': 'date',
                                                'Idpost': 'posts',
                                                'actorpreferredusername': 'unique_users'})

    print(datetime.datetime.now(), 'Summarize VADER')
    vader_monthly = df.pivot_table(
        index=pd.Grouper(key='datetime', freq='M'),
        values=['vader_pos', 'vader_neg', 'vader_compound'],
        aggfunc={
            'vader_pos': 'sum',
            'vader_neg': 'sum',
            'vader_compound': ['mean', 'std']
        }
    ).reset_index()
    vader_monthly.columns = vader_monthly.columns.to_flat_index()

    vader_daily = df.pivot_table(
        index=pd.Grouper(key='datetime', freq='D'),
        values=['vader_pos', 'vader_neg', 'vader_compound'],
        aggfunc={
            'vader_pos': 'sum',
            'vader_neg': 'sum',
            'vader_compound': ['mean', 'std']
        }
    ).reset_index()
    vader_daily.columns = vader_daily.columns.to_flat_index()

    # Columns to extract from final product
    cols = ['date', 'vader_pos_sum', 'vader_neg_sum', 'vader_pos_to_neg_ratio', 'vader_avg_comp', 'vader_sd_comp']

    # VADER pos-to-neg ratio, compound standard deviation
    for dataset in [vader_monthly, vader_daily]:
        # Rename
        dataset.rename(columns={('datetime', ''): 'date',
                                ('vader_pos', 'sum'): 'vader_pos_sum',
                                ('vader_neg', 'sum'): 'vader_neg_sum',
                                ('vader_compound', 'mean'): 'vader_avg_comp',
                                ('vader_compound', 'std'): 'vader_sd_comp'}, inplace=True)

        # Create ratio and SD of VADER compound; reorder columns
        dataset['vader_pos_to_neg_ratio'] = dataset['vader_pos_sum'] / dataset['vader_neg_sum']

    vader_daily['date'] = vader_daily['date'].dt.date
    vader_monthly.rename(columns={'date': 'yearmo'}, inplace=True)
    vader_monthly['yearmo'] = yearmo


    print(datetime.datetime.now(), 'Geo-location statistics')
    state_count = df.groupby('STfips')['STfips'].count()
    state_count = pd.DataFrame({
        'State FIPS': state_count.index,
        'tweets': state_count.values
    })

    total_withST = len(df[~df['STfips'].isna()])
    total_withSTCO = len(df[~df['STCOfips'].isna()])
    geolocated_monthly = pd.DataFrame({
        'yearmo': [yearmo],
        'with State': [total_withST],
        'proportion State': [total_withST / total_tweets],
        'with County': [total_withSTCO],
        'proportion County': [total_withSTCO / total_tweets]
    })

    return_dict = {'volume_monthly': volume_monthly, 'volume_daily': volume_daily,
                   'vader_monthly': vader_monthly, 'vader_daily': vader_daily,
                   'geolocated_monthly': geolocated_monthly, 'count_by_state': state_count,
                   'hashtag': hash_count.head(topn), 'retweet': rt_count.head(topn),
                   'user': user_count.head(topn), 'mention': mention_count.head(topn),
                   'url': url_count.head(topn)}

    return return_dict

# Function to document descriptive statistics
def documentation(descriptives_name, project_name, product_category,
                  author, analytic_path, platform='Twitter', subset='',
                  notes='The descriptive file was generated after removing tweets containing adult websites.'):
    date = datetime.today().strftime('%Y-%m-%d')
    d = {'name': ['Descriptives Name:', 'Project Name:', 'Product Category:', 'Platform:', 'Subset, if Applicable:',
                 'Author:', 'Date Generated:', 'Analytic Data Location:', 'Notes:'],
        'input': [descriptives_name, project_name, product_category,
                  platform, subset, author, date, analytic_path, notes]}
    doc = pd.DataFrame(d)
    return doc

documentation_stat = documentation(
        project_name='FTP R01', product_category='LCC', platform='Twitter', subset='commercial',
        author='SDC',
        analytic_path=fn,
        descriptives_name='{} {} {} descriptives'.format(platform, product_category, subset),
    )


#### PROCESSING #####
data_dir = r'path to twitter data'
raw_fn = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
out_dir = r'path to output folder'
author = 'Haoyu Shi'

emoji_dict = pd.read_excel(r"path to emoji dictionary.xlsx")

for fn in raw_fn:
    print(datetime.datetime.now(), fn)
    yearmo = fn.split('\\')[-1].split('.')[0]

    df = pd.read_parquet(fn)
    #df.replace('nan', None)

    documentation = pd.DataFrame({
        'names': ['analytic_path', 'nrow', 'ncol', 'author'],
        'values': [fn, df.shape[0], df.shape[1], author]
    })

    df['xxx_check'] = df['gnipexpandedurl'].str.contains(domain_pat, case=False, na=False)
    df = df.loc[~df['xxx_check']]

    fix_truncate_rt(df)

    print(datetime.datetime.now(), 'Text Processing')
    fix_truncate_rt(df)
    df['ngram_text'] = df['bodypost'].apply(
        lambda x: text_process(x,
                               remove_hash=True,
                               remove_url=True,
                               remove_punct=True,
                               remove_mention=True,
                               remove_stop=True))
    sentences = [nltk.word_tokenize(sen) for sen in df['ngram_text'].values]
    df_stats = tweet_pd_stats(df)

    print(datetime.datetime.now(), 'emoji statistics')
    # List all emoji in given text column; rename to be informative
    emoji_df = pd.DataFrame(emj.emoji_lis(df['bodypost'].to_string()))
    emoji_df.rename(columns={'location': 'str_location', 'emoji': 'emoji_visualized'}, inplace=True)
    # Escape and decode unicode in new column for emoji's Unicode
    emoji_df['emoji_unicode'] = emoji_df['emoji_visualized'].apply(
        lambda x: x.encode('unicode-escape').decode('ASCII'))
    # Perform demojize on visualized emoji to extract emoji name
    emoji_df['emoji_name'] = emoji_df['emoji_visualized'].apply(lambda x: emj.demojize(x))
    # Value counts
    emoji_output = pd.DataFrame(emoji_df[['emoji_visualized',
                                          'emoji_name', 'emoji_unicode']].value_counts()).reset_index()
    emoji_output.rename(columns={0: 'count'}, inplace=True)

    print('Generating n-grams')
    unigram = make_gram_df(1, sentences)
    bigram = make_gram_df(2, sentences)
    trigram = make_gram_df(3, sentences)

    print(datetime.datetime.now(), 'Saving Monthly Result')
    out_path = os.path.join(out_dir, yearmo + '.xlsx')
    writer = pd.ExcelWriter(out_path, engine='openpyxl')

    documentation.to_excel(writer, sheet_name='documentation', header=False, index=False, encoding='utf-8')

    for k, stat in df_stats.items():
        stat.to_excel(writer, sheet_name=k, index=False, encoding='utf-8')

    emoji_output.head(10000).to_excel(writer, sheet_name='emoji', index=False, encoding='utf-8')
    unigram[0].head(10000).to_excel(writer, sheet_name='unigram', index=False, encoding='utf-8')
    bigram[0].head(10000).to_excel(writer, sheet_name='bigram', index=False, encoding='utf-8')
    trigram[0].head(10000).to_excel(writer, sheet_name='trigram', index=False, encoding='utf-8')

    writer.save()
    writer.close()