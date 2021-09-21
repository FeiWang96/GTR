# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/12/15 15:39
@Description: 
"""
import csv
import json


def process_queries(raw_file):
    with open(raw_file) as rawf, open('train_query.txt', 'w') as trainf, open('dev_query.txt', 'w') as devf, \
            open('test_query.txt', 'w') as testf:
        samples = csv.DictReader(rawf, delimiter='\t')
        for sample in samples:
            if sample['Class'] == 'train':
                f = trainf
            elif sample['Class'] == 'test':
                f = testf
            elif sample['Class'] == 'dev':
                f = devf
            else:
                raise ValueError
            qid = 'web' + sample['QueryID']
            f.write(qid + '\t' + sample['Query'] + '\n')


def process_qtrels(raw_file):
    with open(raw_file) as rawf, open('qtrels.txt', 'w') as qtf:
        samples = csv.DictReader(rawf, delimiter='\t')
        for sample in samples:
            qid = 'web' + sample['QueryID']
            tid = 'web' + sample['TableID']
            qtf.write(qid + '\t' + str(0) + '\t' + tid + '\t' + sample['Label'] + '\n')


def process_tables(raw_file):
    tables = dict()
    with open(raw_file) as rawf:
        samples = csv.DictReader(rawf, delimiter='\t')
        for sample in samples:
            data = [sample['ColumnStr'].split(' _|_ ')]
            data += [row.split(' _|_ ') for row in sample['CellStr'].split(' _||_ ')]

            tid = 'web' + sample['TableID']
            tables[tid] = {
                'caption': sample['Caption'],
                'subcaption': sample['Sub-Caption'],
                'table_array': data
            }

    with open('tables.json', 'w') as tf:
        json.dump(tables, tf)


if __name__ == '__main__':
    process_queries('WebQueryTable_Dataset/WQT.dataset.query.tsv')
    process_qtrels('WebQueryTable_Dataset/WQT.dataset.query-table.tsv')
    process_tables('WebQueryTable_Dataset/WQT.dataset.table.tsv')
