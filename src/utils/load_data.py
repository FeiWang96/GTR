# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/10/14 13:12
@Description: 
"""
import os
import json


def load_tables(data_dir):
    with open(os.path.join(data_dir, 'tables.json')) as f:
        tables = json.load(f)
    return tables


def load_queries(data_dir, file_name='queries.txt'):
    queries = {}
    with open(os.path.join(data_dir, file_name)) as f:
        for line in f.readlines():
            query = line.strip().split()
            queries[query[0]] = query[1:]
    return queries


def load_qt_relations(data_dir):
    qtrels = {}
    with open(os.path.join(data_dir, 'qtrels.txt')) as f:
        for line in f.readlines():
            rel = line.strip().split()
            rel[0] = rel[0]
            rel[3] = int(rel[3])
            if rel[0] not in qtrels:
                qtrels[rel[0]] = {}
            qtrels[rel[0]][rel[2]] = rel[3]
    return qtrels
