# WikiTables Dataset

The dataset is released by [this paper](https://arxiv.org/pdf/1802.06159.pdf).

## Queries and Query-Table Relations

Download queries and query-table relations:
```bash
wget https://raw.githubusercontent.com/iai-group/www2018-table/master/data/queries.txt
wget https://raw.githubusercontent.com/iai-group/www2018-table/master/data/qrels.txt
```
Then remove queries without relevant tables (12, 52 and 53).

## Table Corpus

Download [WikiTables](http://websail-fe.cs.northwestern.edu/TabEL/) corpus:
```bash
wget http://iai.group/downloads/smart_table/WP_tables.zip
unzip WP_tables.zip
```
Extract related tables:
```bash
python extract_table.py
```
Note that:
* we remove the empty tables (table-1452-735 and table-1573-732).
* we combine table['title'] and table['data'] as table['table_array'].