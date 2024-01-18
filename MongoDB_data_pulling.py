# Load Environment variables, required to work with the etl pacakge

from dotenv import load_dotenv
load_dotenv()

# Import MongoQuery object from etl package, this object is written to pull data from MongoDb by Rule or Tag ID
from etl.twitter.db_util import MongoQuery

# Other packages
import os
import pandas as pd
import datetime


if __name__ == '__main__':
    # Specify YYYYMM range for project (08/2016 through 01/2022)
    # ym_list = [str(y) + str(m).zfill(2) for y in range(2016, 2023) for m in range(1, 13)][7:-11]
    ym_list = [str(y) + str(m).zfill(2) for y in range(2019, 2023) for m in range(1, 13)][4:-11] # 201905-202201

    # Extract marijuana rule IDs into list from which data is pulled for FTP
    rules = pd.read_csv(r"path to searching rules.csv")
    mj_rules = rules[rules['Product-Marijuana']==1]['Rule ID'].astype(int).tolist()

    # Folder to save the raw data into
    output_folder = r'path to output folder'

    # Initialize search object
    mongo = MongoQuery('twitter')

    # Loop over the year-month
    for ym in ym_list:
        print(datetime.datetime.now(), ym)
        # Empty list to store tweets from the month
        month_data = []

        for tweet in mongo.search_rule_tag(
                # The field for Rule search is: matchingrulesvalue
                search_field='matchingrulesvalue',
                # The list of Rule IDs
                q_list=mj_rules,
                # The month to pull data from, note that the argument expect a list, i.e. ['202101']
                yearmo_list=[ym]):
            month_data.append(tweet)

        # Turn the list of tweet into a dataframe object
        month_df = pd.DataFrame(month_data)

        # Define output path, a GZIP CSV file
        out_name = os.path.join(output_folder, ym + '.parquet')

        # Save data
        month_df.to_parquet(out_name, index=False)