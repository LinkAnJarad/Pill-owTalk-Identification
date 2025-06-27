# ----------------------------------------

from pymongo import MongoClient
import pandas as pd
from tqdm import tqdm
from google.cloud import secretmanager

def access_secret(secret_id, version_id="latest", project_id="your-gcp-project-id"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

uri = access_secret("mongo-uri", project_id="resolute-casing-464009-s1")

client = MongoClient(uri)
db = client["medications_db"] 

collection_names = ["drug_data", "drug_list", "FDA_pills", "drug_rx"] 

def get_cols(collection):
    sample_doc = collection.find_one()
    cols = [k for k in sample_doc.keys() if k != '_id']
    return cols

def get_data_and_convert_to_df(collection_name):

    df_columns = get_cols(collection=db[collection_name])
    df = {c: [] for c in df_columns}
    df = pd.DataFrame(df)
    rows = [df]
    all_docs = db[collection_name].find()
    # Iterate through documents
    print(f'Fetching {collection_name}...')
    for doc in tqdm(all_docs):
        doc.pop('_id')
        #df = pd.concat([df, pd.DataFrame([doc])])
        rows.append(pd.DataFrame([doc]))
    df = pd.concat(rows, ignore_index=True)
    return df


fda_df = get_data_and_convert_to_df('FDA_pills')
fda_df['INDEX'] = fda_df['INDEX'].astype(int)
drug_list = get_data_and_convert_to_df('drug_list')
drugdata_df = get_data_and_convert_to_df('drug_data')
rx_df = get_data_and_convert_to_df('drug_rx')