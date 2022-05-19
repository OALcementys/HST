from src.python_functions.HST_variations import *
from src.python_functions.database import *
#from python_functions.HST_variations import *
#from python_functions.database import *
import json

"""
        db_host = os.environ['db_host']
        db_name = os.environ['db_name']
        db_user = os.environ['db_user']
        db_port = os.environ['db_port']
        db_password = os.environ['DB_PASSWORD']
        json_region = os.environ['REGION']
"""
def deploy(event,context) :

    try :
        data =event['body']
        print('json file extracted : ', data )
        db_host = os.environ['db_host']
        db_name = os.environ['db_name']
        db_user = os.environ['db_user']
        db_port = os.environ['db_port']
        db_password = os.environ['db_pass']
        queryManager = Query(db_host, db_name, db_user,db_port, db_password)

        res,id_res_var=prepare_res(queryManager,data)
        queryManager.deleteData(id_res_var, '1967-01-01 00:00:00.00+02:00')
        queryManager.db.insertDataframe(res, 'raw_data_save')
        response = {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({
                "variable_id": id_res_var,

            }),
        }


    except:
        response = {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*"
            },
            "body": 'Damaged json file error... pls contact administrator',
        }

    return response


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



event={'info':'info' , body':data}

running locally command  : serverless invoke local --function main --path ./src/test.json

"""
