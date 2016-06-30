
from lcm_postprocess import Timer
from math import log
from multiprocessing import Pool, Process

def f(x):
    z=0
    for y in range(10000):
        z+=y
    return z

def g(x):
    return log(x)

if __name__ == '__main__':

    integers = range(1,1001)

    with Timer() as t:

        squares = [f(x) for x in integers]
        # logs = [g(x) for x in integers]

    print t.interval

    with Timer() as t:

        p = Pool()
        squares = p.map(f, integers)
        # logs = p.map(g, integers)

    print t.interval

    # with Timer() as t:

    #     p = Process(target = f, args = (integers,))
    #     p.start()
    #     p.join()

    #     p = Process(target = g, args = (integers,))
    #     p.start()
    #     p.join()

    # print t.interval