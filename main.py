import os
import struct
import numpy as np
import pandas as pd
from event import event


def main():
    train_events = parsedata(path = 'data\Train')
    test_events = parsedata(path = 'data\Test')
    pass


def parsedata(path):
    events = []
    try:
        i = 0
        for subdir in os.listdir(path):
            subpath = os.path.join(path, subdir)
            for file in os.listdir(subpath):
                file_path = os.path.join(subpath, file)
                e = event(path=file_path, target=i)
                events.append(e)
            i += 1
        return events
    except IOError:
        print(IOError)


if __name__ == '__main__':
    main()
