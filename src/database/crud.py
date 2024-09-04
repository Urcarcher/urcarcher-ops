from src.AI.preprocessing.utils import *
import os

import pymysql

def urcarcher_database_connect() -> pymysql.Connection :
    conn = pymysql.connect(host=os.getenv("BACKEND_URL"), user=os.getenv("DATABASE_CLIENT"), password=os.getenv("DATABASE_CLIENT_PASSWORD"), db=os.getenv("DATABASE"), charset='utf8')
    return conn

def insert_forecasted_exchange_rate(conn :pymysql.Connection, forecasted :pd.DataFrame, exchange_type :str) -> None :
    cur = conn.cursor()
    forecasted = forecasted.to_dict()

    for n in range(len(forecasted['Date'])) :
        sql = f"INSERT INTO forecasted_ex_rate_1yr(forecasted_date, forecasted_open, forecasted_high, forecasted_low, forecasted_close, forecasted_change, exchange_type) VALUES('{convert_to_date_format(forecasted['Date'][n])}', '{forecasted['Open'][n]}', '{forecasted['High'][n]}', '{forecasted['Low'][n]}', '{forecasted['Price'][n]}', '{forecasted['Change %'][n]}', '{exchange_type}')"
        cur.execute(sql)

        print(sql)

    conn.commit()

if __name__ == "__main__" :
    insert_forecasted_exchange_rate(urcarcher_database_connect(), get_exchange_data_from_origins('forecasted_USD_KRW'), 'USD')
