import multiprocessing
import time

def func(x):
    time.sleep(x[0])
    return x[1] + 2

if __name__ == "__main__":    
    p = multiprocessing.Pool()
    start = time.time()
    for x in p.imap_unordered(func, [[1,1],[1,5]]):
        print("{} (Time elapsed: {}s)".format(x, int(time.time() - start)))