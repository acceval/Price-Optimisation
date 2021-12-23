import io
from hashlib import md5

from datetime import datetime, timedelta
from time import localtime
import time
import sys
import inspect
import config



def get_uuid():
    return md5(str(localtime()).encode('utf-8')).hexdigest()


def find(key, dictionary):

    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result


def depth(x):
    if type(x) is dict and x:
        return 1 + max(depth(x[a]) for a in x)
    if type(x) is list and x:
        return 1 + max(depth(a) for a in x)
    return 0

def get_datetime():
    return datetime.now()

def get_today_date():
	return datetime.today().strftime('%Y-%m-%d')

def get_unique_filename(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"

def get_function_caller():
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    return calframe[1][3]

def compress_files(filename, obj):
    fname = str(filename)
    dump(obj, fname, compression="lzma", set_default_extension=False)

def load_compressed_files(filename):    
    fname = str(filename)
    return load(fname, compression="lzma", set_default_extension=False)

def get_date_next_x_days(base,next_x_days):

	base = datetime.strptime(base,'%Y-%m-%d')	
	date_list = [base + timedelta(days=x) for x in range(next_x_days)]
	return [item.strftime('%Y-%m-%d') for item in date_list]

def get_yesterday(base):

    base = datetime.strptime(base,'%Y-%m-%d')   
    return (base - timedelta(days=1)).strftime('%Y-%m-%d')
    

def x_day_diff(base,x):

    base = datetime.strptime(base,'%Y-%m-%d')
    output = base + timedelta(days=x)
    return output.strftime('%Y-%m-%d')

def get_time():

    return time.time()