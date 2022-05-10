import pandas as pd
import psycopg2
from io import StringIO
import os
import pandas as pd
import json
import numpy as np
class Database:
    def __init__(self, host, database, user, password, port):
        self.password = password
        self.host = host
        self.db_name = database
        self.user = user
        self.port = port
        print("Database Ready")
    def getConnection(self):
        conn = psycopg2.connect(host=self.host, dbname=self.db_name, user=self.user, password=self.password, port=self.port)
        return conn
    def selectDataframe(self, query):
        conn = None
        try:
            conn = self.getConnection()
            df = pd.read_sql_query(query, con=conn)
            conn.close()
            return df
        except psycopg2.DatabaseError as error:
            raise error
        finally:
            if conn is not None:
                conn.close()
    def select(self, query, parameter=[]):

        conn = psycopg2.connect(
            host=self.host,
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            port=self.port)
        try:
            cursor = conn.cursor()
            cursor.execute(query, parameter)
        except:
            print("cant execute query")
            raise
        rows = cursor.fetchall()
        conn.close()
        return rows
    def delete(self, query):
        conn = None
        try:
            conn = self.getConnection()
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()
            conn.close()
        except psycopg2.DatabaseError as error:
            raise error
        finally:
            if conn is not None:
                conn.close()
    def insert(self, query, parameter=[]):
        conn = psycopg2.connect(
            host=self.host,
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            port=self.port)
        try:
            cursor = conn.cursor()
            cursor.execute(query, parameter)
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            conn.close()
            return 1
        cursor.close()
        conn.close()
    def insert_returning_id(self, query, parameter=[]):
        conn = psycopg2.connect(
            host=self.host,
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            port=self.port)
        try:
            cursor = conn.cursor()
            cursor.execute(query, parameter)
            id_returning = cursor.fetchone()[0]
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cursor.close()
            conn.close()
            return -1
        cursor.close()
        conn.close()
        return id_returning

    def copy_from_df(self, df):
        conn = psycopg2.connect(
            host=self.host,
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            port=self.port )
        cur = conn.cursor()
        s_buf = io.StringIO()
        df.to_csv(s_buf, header=False, index=True)
        # adapt_row is a custom method, differing from psyopg's adapt
        s_buf.seek(0)
        cur.copy_from(
         s_buf, "raw_data", sep=",", columns=("timestamp", "variable_id", "value") )
        conn.commit()
        cur.close()
        conn.close()

    def insertDataframe(self, df, table):
        # save dataframe to an in memory buffer
        my_df = df.set_index('timestamp')
        buffer = StringIO()
        my_df.to_csv(buffer, index_label='id', header=False)
        buffer.seek(0)

        conn = self.getConnection()
        cursor = conn.cursor()
        #try:
        cursor.copy_from(buffer, table, sep=",", columns=("timestamp", "value", "variable_id"))
        conn.commit()
        #except (Exception, psycopg2.DatabaseError) as error:
        #    print("Error: %s" % error)
        #    conn.rollback()
    #        cursor.close()
            #return 1
        print("Insert dataframe done with success!")
        cursor.close()


    def toPandas(self, query, params = None):
        conn = None
        try:
            conn = self.getConnection()
            data = pd.read_sql(query, conn, params=params)
            return data
        except psycopg2.DatabaseError as error:
            raise error
        finally:
            if conn is not None:
                conn.close()
class Query:

    def __init__(self, host, db_name, user, port, password):
        self.db = Database(
            host=host, database=db_name, user=user, port=port, password=password )

    def get_df_by_variable_id(self, variable_id, variable_name):
        query = """
            SELECT r.timestamp, r.value, r.variable_id FROM raw_data r
            WHERE r.variable_id = %s;
        """
        data = self.db.select(query, (variable_id,))
        df = pd.DataFrame(data=data, columns=["timestamp", variable_name,'id'])
        return df.drop(columns=['id'])

    def get_df_by_variable_id(self, variable_id, variable_name):
        query = """
            SELECT r.timestamp, r.value, r.variable_id FROM raw_data r
            WHERE r.variable_id = %s;
        """
        data = self.db.select(query, (variable_id,))
        df = pd.DataFrame(data=data, columns=["timestamp", variable_name,'id'])
        return df.drop(columns=['id'])

    def get_variable_name_by_id(self, variable_id):

        query="""
        select  variable_id,name from raw_data rd
        join variables v on rd.variable_id=v.id
        WHERE variable_id={0}""".format(variable_id)


        data = self.db.selectDataframe(query)
        return data.iloc[0][1]


    def deleteData(self, offset_variable_id, startDate):
        sqlQuery = """
                    DELETE FROM raw_data_save
                    WHERE timestamp >= to_timestamp('{}','YYYY-MM-DD HH24:MI:SS') and variable_id={}
                    """.format(startDate, offset_variable_id)
        self.db.delete(sqlQuery)


    def get_variable_metric(self, variable_id):
        sqlQuery = """
                        select  unit from variables v join metrics m
                        on v.metric_id =m.id where v.id='{}'

                    """.format(variable_id)
        return self.db.select(sqlQuery)[0][0]



    def insert_result_variable(self, df, res_name, res_metric_id) :
        # create new sensor for the resulted variable
        query = """
                INSERT INTO sensors(name,sensor_type_id,x,y,z,installation_date,initial_frequency,site_gateway_id) values(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id;
            """
        parameter = (res_name,12,1650233,8392540,100,'04/03/2021',20,1)
        sensor_id = self.db.insert_returning_id(query, parameter)
        # print('sensor_id'+ str(sensor_id))
        # creating variables in db for the created sensors
        query = """
                INSERT INTO variables (name,metric_id,sensor_id) values(%s,%s,%s) RETURNING id;
            """
        parameter = (res_name, res_metric_id, sensor_id)
        variable_id = self.db.insert_returning_id(query, parameter)
        # inserting res value to db (raw_data table)
        for i in range (0,len(df)) :
            query = """
                INSERT INTO raw_data (timestamp, value, variable_id) values(%s,%s,%s);
            """
            parameter =  (df.iloc[i]['timestamp'], df.iloc[i]['res'][0], variable_id)
            self.db.insert(query, parameter)

        return variable_id, sensor_id
