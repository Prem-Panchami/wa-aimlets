# This code originated from https://github.com/Prem-Panchami/wa-aimlets
# and is being shared under Creative Commons (4.0) Attribution-Non-Commercial
# license. Please see https://github.com/Prem-Panchami/wa-aimlets/blob/master/LICENSE.md
#
# Retain the above attribution in all copies and derivatives. Thanks!

import numpy as np
import pandas as pd
import collections

def set_defaults():
    pd.set_option('display.width', 110) # Widen the display area so that most tables fit without having to 'fold'

#---------------------------------------------------- .o0o. ----------------------------------------------------#
    
def get_adv_desc_dfs(df, incl_skew=False, incl_kurtosis=False, roundto=2, printout=True, numeric_only=False):
    """Gets advanced descriptions of a Pandas Dataframe.

    Input:
        df =============> Input dataframe
        incl_skew ======> boolean; whether to include skewness in the description
        incl_kurtosis ==> boolean; whether to include kurtosis in the description
        roundto ========> int; number of decimal places to round to when printing.
                          This does NOT affect the return desription dataframes
        printout =======> boolean; whether to print the output before returning
        numeric_only ===> boolean; whether to process numeric columns only
        
    Output:
        Returns 2 dataframes
          - One for numeric columns and the second for non-numeric columns
        
        The first dataframe will contain the following columns:
          - describe() output ===> count  mean   std  min  25%  50%  75%  max
          - skew indicators =====> mean-50%  25%-min  max-75%
          - outlier indicators ==> lwhisk  rwhisk  outliers
          - variance ============> variance
          - skew,kurtosis =======> skew  kurtosis
          
        Note that:
          - lwhisk = 25% - (1.5*IQR) where IQR = 75% - 25%
          - rwhisk = 75% + (1.5*IQR) where IQR = 75% - 25%
     
        The second data frame will contain the following columns:
          - describe() output ===> count unique top freq
    """
    desc_num_df = df.describe(include='number').transpose()
    num_column_names = desc_num_df.index.values
    
#     desc_num_df['_1_'] = ' | '
    
    # mean-50%
    desc_num_df['mean-50%'] = desc_num_df['mean'] - desc_num_df['50%']

    # 25%-min
    desc_num_df['25%-min'] = desc_num_df['25%'] - desc_num_df['min']
    
    # max-75%
    desc_num_df['max-75%'] = desc_num_df['max'] - desc_num_df['75%']
   
    # desc_num_df['_2_'] = ' | '

    # lwhisk = 25% - (1.5*IQR) where IQR = 75% - 25%
    desc_num_df['lwhisk'] = desc_num_df['25%'] - (1.5 * (desc_num_df['75%'] - desc_num_df['25%']))
    
    # rwhisk = 75% + (1.5*IQR) where IQR = 75% - 25%
    desc_num_df['rwhisk'] = desc_num_df['75%'] + (1.5 * (desc_num_df['75%'] - desc_num_df['25%']))
    
    desc_num_df['outliers'] = False
#     desc_num_df['_3_'] = ' | '
    desc_num_df['variance'] = 0.0
    
    if(incl_skew is True):
#         desc_num_df['_4_'] = ' | '
        desc_num_df['skew'] = 0.0
    
    if(incl_kurtosis is True):
        desc_num_df['kurtosis'] = 0.0
    
    for num_col_name in num_column_names:
        if(desc_num_df.loc[num_col_name , 'min'] < desc_num_df.loc[num_col_name, 'lwhisk']
           or
           desc_num_df.loc[num_col_name, 'max'] > desc_num_df.loc[num_col_name, 'rwhisk']):
            desc_num_df.at[num_col_name, 'outliers'] = True

        desc_num_df.at[num_col_name, 'variance'] = df[num_col_name].var()
        
        if(incl_skew is True):
            desc_num_df.at[num_col_name, 'skew'] = df[num_col_name].skew()
    
        if(incl_kurtosis is True):
            desc_num_df.at[num_col_name, 'kurtosis'] = df[num_col_name].kurtosis()

    if(printout is True):
        print("------------------ DATA TYPES AND SHAPE -------------------\n".center(100, ' '))
        print(df.dtypes)
        print()
        print(df.shape)
        print()
        print("--------------------- NUMERIC COLUMNS ---------------------\n".center(100, ' '))
        print(desc_num_df.round(roundto))
        print()

    if(numeric_only is False and 'object' in df.get_dtype_counts().index):
        desc_non_num_df = df.describe(exclude='number').transpose()
        print("------------------- NON-NUMERIC COLUMNS -------------------\n".center(100, ' '))
        print(desc_non_num_df)
        print()
    else:
        desc_non_num_df = None
    
    return desc_num_df, desc_non_num_df

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def get_whiskers_df(df, colList):
    """Gets the whiskers for numeric columns in a Pandas Dataframe.

    Input:
        df ===========> Input dataframe
        colList ======> iterable; list of numeric columns
        
    Output:
        Returns a dataframe with 2 columns (lwhisk, rwhisk) and colList as index

        Note that:
          - lwhisk = 25% - (1.5*IQR) where IQR = 75% - 25%
          - rwhisk = 75% + (1.5*IQR) where IQR = 75% - 25%
     """
    if(not isinstance(colList, collections.Iterable) or isinstance(colList, str)):
        print("raising exception")
        raise Exception("Invalid input for colList - Needs to be list of columns")

    whiskers_df = pd.DataFrame(index=colList, columns=['lwhisk', 'rwhisk'])

    for col in colList:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        whiskers_df.at[col, 'lwhisk'] = q25 - (iqr*1.5)
        whiskers_df.at[col, 'rwhisk'] = q75 + (iqr*1.5)
             
    return whiskers_df
             
#---------------------------------------------------- .o0o. ----------------------------------------------------#

## TODO: Add a donotprint=[list of columns] to handle columns with LOTS of unique values
def get_unique_values_with_counts(df, colList='all'):
    """
    Displays the unique values with their counts given a Pandas Dataframe and a list
    of columns

    Input:
        df ===========> Input dataframe
        colList ======> iterable; list of columns OR 'all' for all columns
        
    Output:
        Returns a dataframe with the following columns:
          - count ===========> DOES NOT include NaN values
          - size ============> Count including NaN values
          - nunique =========> Number of unique values
          - unique_values ===> List of unique values
          - nunique_values ==> List of counts of each unique value
    """
    
    if(isinstance(colList, str)):
        if(colList == 'all'):
            local = df
            print("Using all columns")
        else:
            colList = [colList]
            local = df.loc[:, colList]
            print("Using columns={}".format(local.columns.values))
    elif isinstance(colList, collections.Iterable):
        local = df.loc[:, colList]
        print("Using columns={}".format(local.columns.values))
    else:
        print("raising exception")
        raise Exception("Invalid input for colList - Needs to be list of columns OR 'all'")

    agg = local.agg(['count', 'size', 'nunique']).transpose()
    agg['unique_values'] = ''
    agg['nunique_values'] = ''

    for col in agg.index.values:
        agg.at[col, 'unique_values'], agg.at[col, 'nunique_values'] = np.unique(local[col], return_counts=True)

        print("COLUMN NAME : '{}'".format(col))
        print("  <please note: 'size' includes NaN values, 'count' does not!>")
        print("count".ljust(17,'.') + str(agg.loc[col, 'count']))
        print("size".ljust(17,'.') + str(agg.loc[col, 'size']))
        print("nunique".ljust(17,'.') + str(agg.loc[col, 'nunique']))
        print("unique values".ljust(17,'.') + "{}".format(sorted(zip(agg.loc[col, 'unique_values'], 
                                                           agg.loc[col, 'nunique_values']))))
        print()
        
    return agg

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def strip_whitespaces(df, colList='all'):
    """
    Strips any preceding and following whitespaces from all entries in the given
    list of columns.

    Input:
        df ========> Input dataframe
        colList ===> str or list; 'all', single column name as string or list of
                     column names
        
    Output:
        Returns nothing!
        
        However 'df' will be modified with the removal of whitespaces.
    """
    if(isinstance(colList, str)):
        if(colList == 'all'):
            colList = df.columns.values.tolist()
        else:
            colList = [colList]
    elif not isinstance(colList, collections.Iterable):
        raise Exception("Invalid input for colList - Needs to be list of columns OR 'all'")

    print("strip_whitespaces(): Counts per column:")
    
    for col in colList:
        strip_count = df[col].str.strip().size
        print("{} = {}".format(col, strip_count))
        
    return

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def listify(v):
    """
    Attempts to create a list of items given:
      * A list containing strings, numbers and/or tuples and/or lists
      * A tuple representing the arguments of range() in the same order
      * A string
      * A number
      
    Input:
        v - One of the above items
        
    Output:
        A list if possible
        
        Example: v = [-1, -5, (1,6), [13, 15]] would return the following list:
        [-1, -5, 1, 2, 3, 4, 5, 13, 15]
    """
    from numbers import Number

    retlist = []
    
    if (isinstance(v, list)):
        for item in v:
            if (isinstance(item, tuple)):
                retlist += range(*item)
            elif (isinstance(item, list)):
                retlist += item
            elif (isinstance(item, str)
                or isinstance(item, Number)):
                retlist.append(item)
            else:
                raise Exception("Unsupported data type encountered for {}!!!".format(item))
    elif (isinstance(v, tuple)):
        retlist += range(*v)
    elif (isinstance(v, str)
        or isinstance(v, Number)):
        retlist.append(v)
    else:
        raise Exception("Unsupported data type encountered for {}!!!".format(v))
        
    return retlist

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def resolve_value(strat_dict, col, desc_df):
    
    if(strat_dict[col] == 'mean'):
          return desc_df.loc[col, 'mean']

    elif(strat_dict[col] == 'median'):
          return desc_df.loc[col, '50%']

    elif(strat_dict[col] == 'lwhisk'):
          return desc_df.loc[col, 'lwhisk']

    elif(strat_dict[col] == 'rwhisk'):
          return desc_df.loc[col, 'rwhisk']

    elif(strat_dict[col] == 'whiskers'):
          return (desc_df.loc[col, 'lwhisk'], desc_df.loc[col, 'rwhisk'])

    elif(strat_dict[col] == 'min'):
          return desc_df.loc[col, 'min']

    elif(strat_dict[col] == 'max'):
          return desc_df.loc[col, 'max']

    elif(strat_dict[col][0] == 'fixed'):
          return strat_dict[col][1]

    else:
          raise Exception("Unsupported replacement strategy '{}' specified".format(strat_dict[col]))

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def get_na_cells(df):
    """
    Gets the addresses of all cells that have NA values in numeric columns

    Input:
        df ===========> Input dataframe
        
    Output:
        Returns a dict
          - dict keys : Column names
          - dict values : Indices of cells with NA values
    """
    non_numeric_cols = df.select_dtypes(exclude='number').columns
    if(len(non_numeric_cols) != 0):
        print("WARNING: Ignoring {} non-numeric columns!".format(len(non_numeric_cols), ))
        print(" \n".join(non_numeric_cols))
        print()
 
    na_cols = df.columns[df.isna().any()].tolist()
    
    na_dict = {}
        
    if(len(na_cols) == 0):
        print("GOOD NEWS: No 'na' values were found!")
    else:
        print("!! ALERT !! {} COLUMNS WITH 'na' ENTRIES:".format(len(na_cols)))
        
        for col in na_cols:
            # REFERENCE: https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null
            na_dict[col] = df[df[col].isna()][col].index.values 
            print("'{}' = {}".format(col, len(na_dict[col])))
                
    print()
    return na_dict

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def drop_na_rows(df, reset_index, how='any', thresh=None, subset=None, inplace=True):
    """
    Drops all rows with NA values in a Pandas Dataframe.
    
    This is a wrapper for Dataframe.dropna(). The wrapper is used to fix defaults
    that are commonly used with my projects.

    Input:
        df ==========> Input dataframe that MUST have a column named '_na_'
                       with value 1 for rows with NA values
        
        how =========> {‘any’, ‘all’}, default ‘any’
                       Determine if row is removed from DataFrame, when we have at least one NA or all NA.
                         - ‘any’ : If any NA values are present, drop that row.
                         - ‘all’ : If all values are NA, drop that row.
        
        thresh ======> int, optional
                       Require that many non-NA values.

        subset ======> array-like, optional
                       These would be a list of columns to include when dropping a row

        inplace =====> bool, default True
                       If True, do operation inplace and return None.

        reset_index => bool.
                       If True, the index is reset. False retains old index values after dropping 'na' rows
        Returns:	
        By default returns None.
        If inplace=False, returns DataFrame with NA entries dropped from it
     """

    ret = df.dropna(how=how, thresh=thresh, subset=subset, inplace=inplace)

    if(reset_index is True):
        df.reset_index(drop=True, inplace=True)

    return ret

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def replace_na_entries(df, strat_dict, desc_df=None):
    """
    Replace NA values in numeric columns of a Pandas Dataframe.

    Input:
        df ==========> Input dataframe that MUST have a column named '_na_'
                       with value 1 for rows with NA values
        
        strat_dict ==> dict; this dictionary gives the strategy for replacement
                       values for each column. The keys in 'strat_dict' will be
                       the column names. The values will either be a string or 
                       a tuple. 
                       Valid strings are: 'mean', 'median', 'lwhisk', 'rwhisk', 'min', 'max'
                       Valid tuples are: ('fixed', <some number>)
                                         ('fixed', <some text>)
                       Example: {
                                 'col1' : 'mean',
                                 'col2' : 'median',
                                 'col3' : 'lwhisk',
                                 'col4' : 'rwhisk',
                                 'col5' : ('fixed', 21),
                                 'col6' : ('fixed', 'some fixed text'),
                                 'col7' : 'min',
                                 'col8' : 'max'
                                }
                                 
        desc_df =====> dataframe; each item in colList must be present in the index
                       and that row could have columns 'mean', '50%, 'min', 'max',
                       'lwhisk' and 'rwhisk' with valid values.
                       
                       Pass the 1st returned dataframe from get_adv_desc_dfs()
                       Or if only whiskers are needed, use get_whiskers_df()
                       If None is passed, get_adv_desc_dfs() is called internally.
                               
    Output:
        Returns nothing! 
        
        However 'df' will now contain replaced values for all the columns specified
        in 'strat_dict' as per the specified strategy. 
     """
    if(desc_df is None):
        desc_df, _ = get_adv_desc_dfs(df, printout=False, numeric_only=True)
    
    for col in strat_dict.keys():
        value = resolve_value(strat_dict, col, desc_df)
        
        # It is not guaranteed that the slicing returns a view or a copy. Hence assigning...
        print("Applying replacement strategy '{}' on column='{}'".format(strat_dict[col], col))
        df[col]=df[col].fillna(value)

    return

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def mark_invalid_entries(df, valid_dict, invalid_dict, _invalid_='_invalid_'):
    """
    Mark all rows that have non-valid and invalid entries

    Input:
        df ===========> Input dataframe
                        WARNING: Ensure there is no 'data' column with the reserved
                        name '_invalid_'. This function will modify that column.
                        If you are using the '_invalid_' column to cumulatively track
                        _invalid_ across function calls, include it in 'df'. This 
                        function will update '_invalid_' for any new invalid found.
                        Feel free to change the default invalid column with the 
                        optional _invalid_ parameter
                        
        valid_dict ===> dict; for each column specify the valid or allowed values. Ex:
                        {
                          'col1' : (1, 10), # Tuples for ranges. Values in the same order 
                                              as numpy.arange().
                          'col2' : [2, 4],  # List of numbers
                          'col3' : 2,       # Single number
                          'col4' : 'txt',   # Single string
                          'col5' : ["low", "med", "high"]    # List of strings
                          'col6' : [-1, -5, (1,6), [13, 15]] # List of mixed items. 
                                                               See help for listify()
                        }

        invalid_dict => dict; for each column specify invalid or not-allowed values. Ex:
                        {
                          'col1' : (1, 10), # Tuples for ranges. Values in the same order
                                              as numpy.arange().
                          'col2' : [2, 4],  # List of numbers
                          'col3' : 2,       # Single number
                          'col4' : 'taxt',  # Single string
                          'col5' : '',      # Empty string
                          'col6' : ["law", "mad", "hugh"]      # List of strings
                          'col7' : [-98, 13, (1,6), [97, 101]] # List of mixed items. 
                                                                 See help for listify()
                        }
          
        _invalid_ ====> If you have a different column tracking invalid rows with 0/1, 
                        pass that column name. Default is '_invalid_'. 
                        WARNING: Given that we don't track which column caused the 
                        _invalid_ value to be 1, use this column only for doing row level 
                        operations like dropping rows or invalid row counts.
        
    Output:
        Returns nothing! 
        
        However 'df' will now contain a new _invalid_ column if it wasn't present before.
    """
    
    if(_invalid_ not in df.columns):
        df[_invalid_] = 0
    
    print("Checking items in 'VALID' list:")
    for col in valid_dict.keys():
        print("  BEFORE marking '{}', invalid count = {}".format(col, str(df.loc[df[_invalid_] == 1, _invalid_].count())))
    
        list_of_valid_values = listify(valid_dict[col])
        df.loc[~df[col].isin(list_of_valid_values), _invalid_] = 1

        print("  AFTER  marking '{}', invalid count = {}".format(col, str(df.loc[df[_invalid_] == 1, _invalid_].count())))
        print()

    print()
    print("Checking items in 'INVALID' list:")
    for col in invalid_dict.keys():
        print("  BEFORE marking '{}', invalid count = {}".format(col, str(df.loc[df[_invalid_] == 1, _invalid_].count())))

        list_of_invalid_values = listify(invalid_dict[col])
        df.loc[df[col].isin(list_of_invalid_values), _invalid_] = 1

        print("  AFTER  marking '{}', invalid count = {}".format(col, str(df.loc[df[_invalid_] == 1, _invalid_].count())))
        print()
                         
    return

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def drop_invalid_entries(df, _invalid_='_invalid_'):
    """Drops all invalid rows in a Pandas Dataframe.

    Input:
        df ==========> Input dataframe that MUST have a column named '_invalid_'
                       with value 1 for invalid rows
        
        _invalid_ ===> If you have a different column tracking invalid rows with 0/1, 
                       pass that column name. Default is '_invalid_'

    Output:
        Returns 2 dataframes.
        
          - The first one WITHOUT any row with _invalid_ column value == 1.
          - The second one WILL only contain rows with _invalid_ column value == 1.
          Note: The _invalid_ column is removed in both dataframes.
     """

    if(_invalid_ not in df.columns):
        raise Exception("Please make sure there is a column named '{}' with \
                    value 1 for invalid rows. Please see drop_invalid_entries()...".format(_invalid_))

    return df.loc[df[_invalid_] == 0, df.columns != _invalid_], \
              df.loc[df[_invalid_] == 1, df.columns != _invalid_]
              
#---------------------------------------------------- .o0o. ----------------------------------------------------#


def replace_non_valid_entries(df, valid_dict, strat_dict, desc_df=None):
    """
    Any value that is NOT IN the valid list, replace with a single value

    Input:
        df ==========> Input dataframe
        
        valid_dict ==> dict; for each column specify the valid or allowed values. Ex:
                        {
                          'col1' : (1, 10), # Tuples for ranges. Values in the same order 
                                              as numpy.arange().
                          'col2' : [2, 4],  # List of numbers
                          'col3' : 2,       # Single number
                          'col4' : 'txt',   # Single string
                          'col5' : ["low", "med", "high"]    # List of strings
                          'col6' : [-1, -5, (1,6), [13, 15]] # List of mixed items. 
                                                               See help for listify()
                        }

        strat_dict ==> dict; this dictionary gives the strategy for replacement
                       values for each column. The keys in 'strat_dict' will be
                       the column names. The values will either be a string or 
                       a tuple. 
                       Valid strings are: 'mean', 'median', 'lwhisk', 'rwhisk', 'min', 'max'
                       Valid tuples are: ('fixed', <some number>)
                                         ('fixed', <some text>)
                       Example: {
                                 'col1' : 'mean',
                                 'col2' : 'median',
                                 'col3' : 'lwhisk',
                                 'col4' : 'rwhisk',
                                 'col5' : ('fixed', 21),
                                 'col6' : ('fixed', 'some fixed text'),
                                 'col7' : 'min',
                                 'col8' : 'max'
                                }
                                 
        desc_df =====> dataframe; each item in colList must be present in the index
                       and that row could have columns 'mean', '50%, 'min', 'max',
                       'lwhisk' and 'rwhisk' with valid values.
                       
                       Pass the 1st returned dataframe from get_adv_desc_dfs()
                       Or if only whiskers are needed, use get_whiskers_df()
                       If None is passed, get_adv_desc_dfs() is called internally.

    Output:
        Return nothing!
        
        However 'df' will be modified based on the replacement strategy given.
    """
    if(desc_df is None):
        desc_df, _ = get_adv_desc_dfs(df, printout=False, numeric_only=True)
    
    for col in strat_dict.keys():
        value = resolve_value(strat_dict, col, desc_df)

        list_of_valid_values = listify(valid_dict[col])
        df.loc[~df[col].isin(list_of_valid_values), col] = value
    
    return

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def replace_invalid_column_entries(df, colName, invalid_list, replace_with, desc_df=None):
    """
    For a given column, any value that is IN the invalid list will be replaced 
    with a single value

    Input:
        df ===========> Input dataframe
        
        colName ======> str; name of the column
        
        invalid_list => list, tuple, number, str; Specify values to be replaced. Ex:
                        (1, 10), # Tuples for ranges. Values in the same order 
                                   as numpy.arange().
                        [2, 4],  # List of numbers
                        2,       # Single number
                        'txt',   # Single string
                        ["low", "med", "high"]    # List of strings
                        [-1, -5, (1,6), [13, 15]] # List of mixed items. 
                                                    See help for listify()

        replace_with => string or tuple;
                       Valid strings are: 'mean', 'median', 'lwhisk', 'rwhisk', 'min', 'max'
                       Valid tuples are: ('fixed', <some number>) Ex: ('fixed', 21)
                                         ('fixed', <some text>) Ex: ('fixed', 'some fixed text')
                                 
        desc_df =====> dataframe; each item in colList must be present in the index
                       and that row could have columns 'mean', '50%, 'min', 'max',
                       'lwhisk' and 'rwhisk' with valid values.
                       
                       Pass the 1st returned dataframe from get_adv_desc_dfs()
                       Or if only whiskers are needed, use get_whiskers_df()
                       If None is passed, get_adv_desc_dfs() is called internally.

    Output:
        Return nothing!
        
        However 'df' will be modified based on the replacement strategy given.
    """
    if(desc_df is None):
        desc_df, _ = get_adv_desc_dfs(df, printout=False, numeric_only=True)
    
    value = resolve_value({colName : replace_with}, colName, desc_df)

    list_of_invalid_values = listify(invalid_list)
    df.loc[df[colName].isin(list_of_invalid_values), colName] = value
              
    return

#---------------------------------------------------- .o0o. ----------------------------------------------------#
## TODO: Take another parameter that clears out the _outlier_ column. Useful for iteratively replacing and checking
def mark_outliers(df, colList, whiskers_df=None, _outlier_='_outlier_'):
    """
    Mark all rows that have outliers in numeric columns

    Input:
        df ===========> Input dataframe
                        WARNING: Ensure there is no 'data' column with the reserved
                        name '_outlier_'. This function will modify that column.
                        If you are using the '_outlier_' column to cumulatively track
                        outliers across function calls, include it in 'df'. This 
                        function will update '_outlier_' for any new outliers found.
                        Feel free to change the default outlier column with the 
                        optional _outlier_ parameter
                        
        colList ======> iterable; list of numeric columns
        
        whiskers_df ==> dataframe; each item in colList must be present in the index
                        and that row must have columns 'lwhisk' and 'rwhisk' with 
                        valid values. One can simply pass the 1st returned dataframe
                        from get_adv_desc_dfs().
                        
                        If None is passed, this will be automatically calculated.
                        
        _outlier_ ====> If you have a different column tracking outliers with 0/1, pass
                        that column name. Default is '_outlier_'. 
                        WARNING: Given that we don't track which column caused the 
                        _outlier_ value to be 1, use this column only for doing row level 
                        operations like dropping rows or outlier row counts.
        
    Output:
        Returns nothing! 
        
        However 'df' will now contain a new _outlier_column if it wasn't present before.
    """
    if(not isinstance(colList, collections.Iterable) or isinstance(colList, str)):
        print("raising exception")
        raise Exception("Invalid input for colList - Needs to be list of columns")

    print("Using columns={}\n".format(colList))

    if(whiskers_df is None):
        whiskers_df = get_whiskers_df(df, colList)

    if(_outlier_ not in df.columns):
        df[_outlier_] = 0
    
    for col in colList:
        print("BEFORE marking '{}', outlier count = {}".format(col, str(df.loc[df[_outlier_] == 1, _outlier_].count())))
    
        df.loc[df[col] < whiskers_df.loc[col, 'lwhisk'], _outlier_] = 1
        df.loc[df[col] > whiskers_df.loc[col, 'rwhisk'], _outlier_] = 1
    
        print("AFTER  marking '{}', outlier count = {}".format(col, str(df.loc[df[_outlier_] == 1, _outlier_].count())))
        print()

    return

#---------------------------------------------------- .o0o. ----------------------------------------------------#

def drop_outliers(df, _outlier_='_outlier_'):
    """Drops all rows with outliers in a Pandas Dataframe.

    Input:
        df ==========> Input dataframe that MUST have a column named '_outlier_'
                       with value 1 for outlier rows
        
        _outlier_ ===> If you have a different column tracking outliers with 0/1, pass
                       that column name. Default is '_outlier_'

    Output:
        Returns 2 dataframes.
        
          - The first one WITHOUT any row with _outlier_ column value == 1.
          - The second one WILL only contain rows with _outlier_ column value == 1.
          Note: The _outlier_ column is removed in both dataframes.
     """

    if(_outlier_ not in df.columns):
        raise Exception("Please make sure there is a column named '{}' with \
                    value 1 for outliers. Please see mark_outliers()...".format(_outlier_))

    return df.loc[df[_outlier_] == 0, df.columns != _outlier_], \
              df.loc[df[_outlier_] == 1, df.columns != _outlier_]
              
#---------------------------------------------------- .o0o. ----------------------------------------------------#
         
def replace_outliers(df, strat_dict, desc_df=None):
    """Replace outliers in numeric columns of a Pandas Dataframe.

    Input:
        df ==========> Input dataframe
        
        strat_dict ==> dict; this dictionary gives the strategy for replacement
                       values for each column. The keys in 'strat_dict' will be
                       the column names. The values will either be a string or 
                       a tuple. 
                       Valid strings are: 'mean', 'median', 'whiskers', 'min', 'max'
                       Valid tuples are: ('fixed', <some number>)
                                         ('fixed', <some text>)
                       Example: {
                                 'col1' : 'mean',
                                 'col2' : 'median',
                                 'col3' : 'whiskers',
                                 'col4' : ('fixed', 21),
                                 'col5' : ('fixed', 'some fixed text'),
                                 'col6' : 'min',
                                 'col7' : 'max'
                                }
                                 
                       Note: All values in that column < lwhisk and > rwhisk will be 
                       replaced.
                                 
        desc_df =====> dataframe; each item in colList must be present in the index
                       and that row could have columns 'mean', '50%, 'min', 'max',
                       'lwhisk' and 'rwhisk' with valid values.
                       
                       Pass the 1st returned dataframe from get_adv_desc_dfs()
                       Or if only whiskers are needed, use get_whiskers_df()
                       If None is passed, get_adv_desc_dfs() is called internally.
        
    Output:
        Returns nothing! 
        
        However 'df' will now contain replaced values for all the columns specified
        in 'strat_dict' as per the specified strategy.
     """

    if(desc_df is None):
        desc_df, _ = get_adv_desc_dfs(df, printout=False, numeric_only=True)
    
    for col in strat_dict.keys():
        lvalue = resolve_value(strat_dict, col, desc_df)
        rvalue = lvalue
        
        if(isinstance(lvalue, tuple)):
              rvalue = lvalue[1]
              lvalue = lvalue[0]
         
        print("Applying replacement strategy '{}' on column='{}'".format(strat_dict[col], col))
        df.loc[df[col] < desc_df.loc[col, 'lwhisk'], col] = lvalue
        df.loc[df[col] > desc_df.loc[col, 'rwhisk'], col] = rvalue

    return
              
#---------------------------------------------------- .o0o. ----------------------------------------------------#

import sklearn.metrics as metrics

def print_scores(x_tst, y_tst, y_pred, average='binary'):    
    print("SCORES:")
    print(" 'metrics.accuracy_score'........ {}".format(metrics.accuracy_score(y_tst, y_pred)))
    print(" 'metrics.f1_score'.............. {}".format(metrics.f1_score(y_tst, y_pred, average=average)))
    print()
    print("CONFUSION MATRIX:")
    print(metrics.confusion_matrix(y_tst, y_pred))
    
    print("\n Accuracy : Overall, how often is the classifier correct?")
    print(" 'metrics.accuracy_score'........ {}".format(metrics.accuracy_score(y_tst, y_pred)))

    print("\n Sensitivity or Recall : When it's actually yes, how often does it predict yes?")
    print(" 'metrics.recall_score'.......... {}".format(metrics.recall_score(y_tst, y_pred, average=average)))
    
    print("\n Precision : When it predicts yes, how often is it correct?")
    print(" 'metrics.precision_score'....... {}".format(metrics.precision_score(y_tst, y_pred, average=average)))
    
    print()
    print("CLASSIFICATION REPORT:")
    print(metrics.classification_report(y_tst, y_pred))

    print("--------------------------------------------------------")

#---------------------------------------------------- .o0o. ----------------------------------------------------#
# get_correlation
# chi2_contingency p-values