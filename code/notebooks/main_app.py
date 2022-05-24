#!/usr/bin/env python
# coding: utf-8

# In[1]:


from database import *
import psycopg2
from HST_variations import *
from psycopg2.extras import register_json
import pandas as pd
from io import StringIO
import os
import pandas as pd
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Deploy arguments')

    parser.add_argument('--dbHost', type=str, default='13.36.221.162',
                        help='database host')
    parser.add_argument('--dbName', type=str, default='thminsight',
                            help='database name')
    parser.add_argument('--dbUser', type=str, default='thminsight',
                                                        help='databse user ')
    parser.add_argument('--dbPort', type=str, default='5432',
                                help='databse user  ')
    parser.add_argument('--dbPassword', type=str, default='psS8hqr6oaoJ7fgk',
                        help='databse Password ')
    parser.add_argument('--json_path', type=str, default='.json_path.json',
                        help='path to json file model details')

    opt = parser.parse_args()
    return opt

def lambda_handler(event,context opt) :
    json_region = os.environ['AWS_REGION']
    try :
        #data = json.loads(event['body'])
        dbHost =opt.dbHost
        dbName = opt.dbName
        dbUser = opt.dbUser
        dbPort =opt.dbPort
        dbPassword = opt.dbPassword
        queryManager = Query(dbHost, dbName, dbUser,dbPort, dbPassword)

        #deploy method
        res,id_res_var=prepare_res(queryManager,data)
        queryManager.deleteData(id_res_var, '1967-01-01 00:00:00.00+02:00')
        queryManager.db.insertDataframe(df_res, 'raw_data_save')



        response = {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({
                "variable_id": res_id[0],

            }),
        }
        print(response)
        return response
    except:
        response = {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": 'Damaged json file error... pls contact administrator',
        }
