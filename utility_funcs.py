import numpy as np
import datetime


def get_date(date_str, format_str='%Y-%m-%d %H:%M:%S'):
    if type(date_str) == np.ndarray:
        date_str = date_str[0]
    if type(date_str) == datetime.datetime:
        return date_str
    elif type(date_str) == float:
        return 'nan'
    try:
        date_str = date_str.decode("utf-8").split('.')[0]
    except AttributeError:
        date_str = date_str.split('.')[0]
    try:
        date = datetime.datetime.strptime(date_str, format_str)
    except ValueError:
        format_str = '%m/%d/%y %H:%M'
        try:
            date = datetime.datetime.strptime(date_str, format_str)
        except ValueError:
            format_str = '%m/%d/%y'
            try:
                date = datetime.datetime.strptime(date_str, format_str)
            except ValueError:
                format_str = '%m/%d/%Y'
                try:
                    date = datetime.datetime.strptime(date_str, format_str)
                except ValueError:
                    return 'nan'
    return date


def get_array_dates(array, date_fmt='%Y-%m-%d %H:%M:%S'):
    out = np.zeros(array.shape, dtype=np.object)
    for i in range(int(np.prod(array.shape))):
        idx = np.unravel_index(i, array.shape)
        out[idx] = get_date(array[idx], format_str=date_fmt)
    return out