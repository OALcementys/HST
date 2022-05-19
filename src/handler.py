from src.HST_variations import *
from src.database import *
import json



def deploy(event,context) :
    json_region = os.environ['AWS_REGION']
    try :
        data = json.loads(event['body'])
        db_host = os.environ['DB_HOST']
        db_name = os.environ['DB_NAME']
        db_user = os.environ['DB_USER']
        db_port = os.environ['DB_PORT']
        db_password = os.environ['DB_PASSWORD']
        queryManager = Query(dbHost, dbName, dbUser,dbPort, dbPassword)
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
                "variable_id": res_id,

            }),
        }
        return response
    except:
        response = {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": 'Damaged json file error... pls contact administrator',
        }
    return None


""""
json file form example
data={'model': [{'target': 1238,
   'H': 1246,
   'T': True,
   'S': True,
   'temp': 1245,
   'V': 1238},
  {'Approch': 'use_NN_optimizer',
   'Optimizer': 'SGD',
   'loss': 'MSE',
   'metric': 'R2',
   'interpolation': 'upsampling',
   'res_id': 112}]}



event={'body':data}
"""
