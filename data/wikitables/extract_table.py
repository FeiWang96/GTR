# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/10/13 21:42
@Description: 
"""
import os
import json

RAW_TABLE_DIR = '../retrieval_data/tables_redi2_1'
QTREL_FILE = 'qrels.txt'

table_ids = set()
qtables = {}


def get_table_ids():
    with open(QTREL_FILE) as f:
        for line in f.readlines():
            table_ids.add(line.strip().split()[2])
    print(len(table_ids), 'tables in total')


def collect_tables():
    for file in os.listdir(RAW_TABLE_DIR):
        with open(os.path.join(RAW_TABLE_DIR, file)) as f:
            tables = json.load(f)
            for idx, table in tables.items():
                if idx in table_ids:
                    if convert_table(table):
                        qtables[idx] = table
                    else:
                        print('error:', idx)
                        print(table)

    with open('tables.json', 'w') as f:
        json.dump(qtables, f)


def convert_table(table):
    if (len(table['data']) == 0 and len(table['title']) == 0) or (len(table['data']) != 0 and len(table['title']) != len(table['data'][0])):
        return False
    table['table_array'] = [table['title']] + table['data']
    return True


if __name__ == '__main__':
    get_table_ids()
    collect_tables()
