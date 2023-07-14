#!/usr/bin/env python

import argparse
import json
import os
from os.path import dirname, realpath

from pyspark import SparkContext


def parse_args():
    parser = argparse.ArgumentParser(description='MapReduce join (Problem 2)')
    parser.add_argument('-d', help='path to data file', default='./../data/records.json')
    parser.add_argument('-n', help='number of data slices', default=128)
    parser.add_argument('-o', help='path to output JSON', default='output')
    return parser.parse_args()


# Feel free to create more mappers and reducers.
def mapper(record):
    table_name = record[0]
    company_id = record[2]
    return(table_name, [record])

def reducer1(a, b):
    return a + b

#def mapper2():
    #pass

def reducer2(a, b):
    result = []
    for row_a in a[1]:
        for row_b in b[1]:
            if row_a[2] == row_b[2]:
                result.append(row_b + row_a)
    return result



def main():
    args = parse_args()
    sc = SparkContext()

    with open(args.d, 'r') as infile:
        data = [json.loads(line) for line in infile]

    # TODO: build your pipeline
    join_result = sc.parallelize(data, 128).map(mapper).reduceByKey(reducer1).reduce(reducer2)

    sc.stop()

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    with open(args.o + '/output_join.json', 'w') as outfile:
        json.dump(join_result, outfile, indent=4)


if __name__ == '__main__':
    main()
