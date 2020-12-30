import linecache
import os
import yaml
import random
import tracemalloc
import sqlite3
import numpy as np
from sqlite3 import Error


def create_database(filepath):
    database = filepath
    sql_create_indexes_table = """ CREATE TABLE IF NOT EXISTS indexes (
                                    id integer PRIMARY KEY,
                                    row_id integer NOT NULL,
                                    selected_index text NOT NULL
                                ); """

    conn = create_connection(database)
    if conn is not None:
        create_table(conn, sql_create_indexes_table)
    else:
        print("Error! cannot create the database connection.")
    return conn


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def add_index(conn, row_id, array):
    """
    Convert numpy array to bytes and insert into database
    :param conn: SQLite Connection Object
    :param array: A numpy array
    :return: id of last row entered
    """
    asbyte = array.dumps()
    sql = ''' INSERT INTO indexes(row_id, selected_index)
              VALUES(?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, (row_id, asbyte))
    conn.commit()
    return cur.lastrowid

