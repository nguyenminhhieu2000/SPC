import curses
from distutils.util import execute
import os
import sqlite3 as lite
from sqlite3 import Error


#===========================================================================
def create_info_default(path):
    '''
    Create username and password default for app.
    '''
    con = lite.connect(path)
    login_info = (
        ('admin', 'admin', 'developer'),
        ('nurse1', 'nurse1', 'nurse'),
        ('patient1', 'patient1', 'patient')
    )
    with con:
        cur = con.cursor()
        cur.execute('drop table if exists login_info')
        cur.execute('create table login_info(username text, password text, role text)')
        cur.executemany('insert into login_info values(?, ?, ?)', login_info)

#===========================================================================
def create_connection(path_db):
    '''
    Function to check file db exists or not. If not create a new db file.
    '''
    file_exist = os.path.exists(path_db)
    if file_exist:
        print('Database has created!')
    else:
        print('Create database file.')
        con = None
        try:
            con = lite.connect(path_db)
            print(lite.version)
        except Error as e:
            print(e)
        finally:
            if con:
                con.close()
        
        create_info_default(path_db)