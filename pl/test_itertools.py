import itertools


def count(n = 0):
    '''n'''
    while True:
        yield n
        n += 1

iter_var = count()

print(itertools.islice(iter_var, 3, None))
print(next(itertools.islice(iter_var, 3, None)))