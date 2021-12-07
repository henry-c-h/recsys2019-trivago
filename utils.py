import numpy as np
import logging
from pandas.api.types import CategoricalDtype


# adapted from
# https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df, logger):
    logger.info('Reducing memory usage...')

    start_mem_usg = df.memory_usage().sum() / 1024**2
    logger.info(f'Memory usage before reducing: {int(start_mem_usg)} MB.')
    for col in df.columns:
        # if df[col].dtype != object:  # Exclude strings
        if df[col].dtype != object and not isinstance(df[col].dtype, CategoricalDtype):
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()

            # test if column can be converted to an integer
            if df[col].isnull().sum() == 0:
                asint = df[col].astype(np.int64)
                result = (df[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
         
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
   
    # Print final result
    mem_usg = df.memory_usage().sum() / 1024**2 
    logger.info(f'Memory usage after reducing: {int(mem_usg)} MB. '\
          f'Memory reduced by {int((start_mem_usg - mem_usg)/start_mem_usg*100)}%.')
    
    return df


def safe_divide(s1, s2):
    res = np.where(s2!=0, s1/s2, 0)
    return res


def create_logger(file_name='recsys.log', level=logging.INFO):
    logger = logging.getLogger('RecSys Trivago Challenge')
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(file_name, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

