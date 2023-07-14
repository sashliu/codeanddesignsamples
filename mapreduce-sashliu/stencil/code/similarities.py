#!/usr/bin/env python

from __future__ import division
import argparse
import json
import math
from os.path import dirname, realpath
from pyspark import SparkContext
import time
import os

VIRTUAL_COUNT = 10
PRIOR_CORRELATION = 0.0
THRESHOLD = 0.5


##### Metric Functions ############################################################################
def correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy):
    # http://en.wikipedia.org/wiki/Correlation_and_dependence
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt(n * sum_xx - sum_x * sum_x) * math.sqrt(n * sum_yy - sum_y * sum_y)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def regularized_correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy, virtual_count, prior_correlation):
    unregularized_correlation_value = correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
    weight = n / (n + virtual_count)
    return weight * unregularized_correlation_value + (1 - weight) * prior_correlation

def cosine_similarity(sum_xx, sum_yy, sum_xy):
    # http://en.wikipedia.org/wiki/Cosine_similarity
    numerator = sum_xy
    denominator = (math.sqrt(sum_xx) * math.sqrt(sum_yy))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def jaccard_similarity(n_common, n1, n2):
    # http://en.wikipedia.org/wiki/Jaccard_index
    numerator = n_common
    denominator = n1 + n2 - n_common
    if denominator == 0:
        return 0.0
    return numerator / denominator
#####################################################################################################

##### util ##########################################################################################
def combinations(iterable, r):
    # http://docs.python.org/2/library/itertools.html#itertools.combinations
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(list(range(r))):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)
#####################################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='MapReduce similarities')
    parser.add_argument('-d', help='path to data directory', default='./../data/recommendations/small/')
    parser.add_argument('-n', help='number of data slices', default=128)
    parser.add_argument('-o', help='path to output JSON', default="output")
    return parser.parse_args()

# Feel free to create more mappers and reducers.
def mapper0(record):
    items = record.split('::')
    if len(items) == 3:
        return(items[1], [items[0], items[2]])
    else:
        return (items[0], [items[1]])

def reducer(a, b):
    return a + b

def mapper1(record):
    # Hint:
    # INPUT:
    #   record: (key, values)
    #     where -
    #       key: movie_id
    #       values: a list of values in the line
    # OUTPUT:
    #   [(key, value), (key, value), ...]
    #     where -
    #       key: movie_title
    #       value: [(user_id, rating)]
    #
    # TODO
    for i in range(1, len(record[1]), 2):
        yield (record[1][0], ([[record[1][i], int(record[1][i+1])]]))

def mapper2(record):
    for review in record[1]:
        yield (review[0], [[record[0], review[1], len(record[1])]])

def mapper3(record):
    for movie1 in record[1]:
        for movie2 in record[1]:
            if movie1[0] != movie2[0] and movie1[0] < movie2[0]:
                yield((movie1[0], movie2[0]), [[[movie1[1], movie1[2]], [movie2[1], movie2[2]]]])

def mapper4(record):
    n = len(record[1])
    n1 = record[1][0][0][1]
    n2 = record[1][0][1][1]
    sum_x = 0
    sum_y = 0
    sum_xx = 0
    sum_yy = 0
    sum_xy = 0
    for pair in record[1]:
        sum_x += pair[0][0]
        sum_y += pair[1][0]
        sum_xx += (pair[0][0])**2
        sum_yy += (pair[1][0])**2
        sum_xy += (pair[0][0])*(pair[1][0])
    movie_correlation = correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy)
    reg_correlation = regularized_correlation(n, sum_x, sum_y, sum_xx, sum_yy, sum_xy, 10, 0)
    cosine_sim = cosine_similarity(sum_xx, sum_yy, sum_xy)
    jaccard_sim = jaccard_similarity(n, n1, n2)
    if reg_correlation > 0.5:
        return[(record[0][0], [(record[0][1], movie_correlation, reg_correlation, cosine_sim, jaccard_sim, n, n1, n2)])]
    else:
        return[]

def mapper5(record):
    for movie in record[1]:
        yield((record[0], movie[0]), (movie[1], movie[2], movie[3], movie[4], movie[5], movie[6], movie[7]))

def main():
    args = parse_args()
    sc = SparkContext()

    with open(args.d + '/movies.dat', 'r') as mlines:
        data = [line.rstrip() for line in mlines]
    with open(args.d + '/ratings.dat', 'r') as rlines:
        data += [line.rstrip() for line in rlines]

    # FEEL FREE TO EDIT ANY OF THE MAPPER/REDUCER FUNCTION CALLS, BUT ENSURE
    # THE STAGE 1, STAGE 2, and FINAL RESULTS MATCH THE EXPECTED STRUCTURE

    # Implement your mapper and reducer function according to the following query.
    # stage1_result represents the data after it has been processed at the second
    # step of map reduce, which is after mapper1.
    stage1_result = sc.parallelize(data, args.n).map(mapper0).reduceByKey(reducer) \
                                                    .flatMap(mapper1).reduceByKey(reducer)
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # Store the stage1_output
    with open(args.o  + '/netflix_stage1_output.json', 'w') as outfile:
        json.dump(stage1_result.collect(), outfile, separators=(',', ':'))

    # TODO: continue to build the pipeline
    # Pay attention to the required format of stage2_result
    stage2_result = stage1_result.flatMap(mapper2).reduceByKey(reducer).flatMap(mapper3).reduceByKey(reducer).flatMap(mapper4).reduceByKey(reducer)

    # Store the stage2_output
    with open(args.o  + '/netflix_stage2_output.json', 'w') as outfile:
        json.dump(stage2_result.collect(), outfile, separators=(',', ':'))

    # TODO: continue to build the pipeline
    final_result = stage2_result.flatMap(mapper5).collect()

    with open(args.o + '/netflix_final_output.json', 'w') as outfile:
        json.dump(final_result, outfile, separators=(',', ':'))

    sc.stop()



if __name__ == '__main__':
    main()
