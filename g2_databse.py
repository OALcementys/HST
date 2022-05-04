import psycopg2
from psycopg2.extras import register_json
import pandas as pd

class Database:
    def __init__(self, host, database, user, password, port):
        self.password = password
        self.host = host
        self.db_name = database
        self.user = user
        self.port = port
        register_json(oid=3802, array_oid=3807)

    # get connection , to perform transactions outside this component
    def getConnection(self):
        conn = psycopg2.connect(host=self.host, dbname=self.db_name, user=self.user, password=self.password, port=self.port)
        return conn

    # select query
    def select(self, query, parameters=[]):
        conn = None
        try:
            conn = self.getConnection()
            cursor = conn.cursor()
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            return rows
        except psycopg2.DatabaseError as error:
            raise error
        finally:
            if conn is not None:
                conn.close()

    # insert query
    def insert(self, query):
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
    def pyquery(self,query,join=False):
        conn = None
        try:
            conn = self.getConnection()
            cursor = conn.cursor()
            cursor.execute(query)
            column_names = [desc[0] for desc in cursor.description]
            table = cursor.fetchall()
            cursor.close()
            conn.close()
            if join : 
                return pd.DataFrame(table , columns=column_names)
            else: 
                return pd.DataFrame(table)
        except psycopg2.DatabaseError as error:
            raise error
        finally:
            if conn is not None:
                conn.close()

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
