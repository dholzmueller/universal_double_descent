import os
import os.path
import heapq
import glob
import gzip
import shutil
import pickle
import copy


def select_from_config(config, keys):
    selected = {}
    for key in keys:
        if key in config:
            selected[key] = config[key]
    return selected

def adapt_config(config, **kwargs):
    new_config = copy.deepcopy(config)
    for key, value in kwargs.items():
        new_config[key] = value
    return new_config

def existsDir(directory):
    if directory != '':
        if not os.path.exists(directory):
            return False
    return True

def existsFile(file_path):
    return os.path.isfile(file_path)

def ensureDir(file_path):
    directory = os.path.dirname(file_path)
    if directory != '':
        if not os.path.exists(directory):
            os.makedirs(directory)

def matchFiles(file_matcher):
    return glob.glob(file_matcher)

def newDirname(prefix):
    i = 0
    name = prefix
    if existsDir(prefix):
        while existsDir(prefix + "_" + str(i)):
            i += 1
        name = prefix + "_" + str(i)
    os.makedirs(name)
    return name

def getSubfolderNames(folder):
    return [os.path.basename(name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]

def getSubfolders(folder):
    return [os.path.join(folder, name)
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]


def writeToFile(filename, content):
    ensureDir(filename)
    file = open(filename, 'w')
    file.truncate()
    file.write(content)
    file.close()


def readFromFile(filename):
    if not os.path.isfile(filename):
        return ''

    file = open(filename, 'r')
    result = file.read()
    file.close()
    return result


def serialize(filename, obj, compressed=False):
    ensureDir(filename)
    if compressed:
        file = gzip.open(filename, 'wb')
    else:
        file = open(filename, 'wb')
    pickle.dump(obj, file, protocol=3)
    file.close()

def deserialize(filename, compressed=False):
    if compressed:
        file = gzip.open(filename, 'rb')
    else:
        file = open(filename, 'rb')
    result = pickle.load(file)
    file.close()
    return result

def copyFile(src, dst):
    ensureDir(dst)
    shutil.copyfile(src, dst)

def nsmallest(n, inputList):
    return heapq.nsmallest(n, inputList)[-1]

def identity(x):
    return x

def set_none_except(lst, idxs):
    for i in range(len(lst)):
        if i not in idxs:
            lst[i] = None

def argsort(lst):
    # from https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    return sorted(range(len(lst)), key=lst.__getitem__)
