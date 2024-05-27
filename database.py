import pymysql

def get_db_connection():
    return pymysql.connect(host='127.0.0.1',
                           user='root',
                           password='root',
                           db='3dscan_db')

# connection = pymysql.connect(host='localhost',
#                                 user='c203dscan_api',
#                                 password='Fvg4!3wHN',
#                                 db='c203dscan_db_api')

