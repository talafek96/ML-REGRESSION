from prepare2 import prepare_data
from sklearn.model_selection import train_test_split
import pandas as pd
import inspect


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():
    dataset = pd.read_csv('csv/virus_labeled.csv')
    train, test = train_test_split(dataset, test_size=0.2, random_state=10)
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    print(f'Train shape is {train.shape} and test shape is {test.shape}')

    train_clean = prepare_data(train, train)
    test_clean = prepare_data(test, train)
    print(
        f'New train shape is {train_clean.shape} and new test shape is {test_clean.shape}')

    print(f'Train head:\n{train_clean.head(10)}\n\n')
    print(f'Test head:\n{test_clean.head(10)}\n\n')
    check_na(train_clean)
    check_na(test_clean)

    # Export to csv files:
    train_path = 'csv/virus_train.csv'
    test_path = 'csv/virus_test.csv'
    train_clean.to_csv(train_path, index=False)
    test_clean.to_csv(test_path, index=False)


def check_na(df: pd.DataFrame):
    this_function_name = inspect.currentframe().f_code.co_name

    failed = False
    errors = str()
    for column in df:
        df_col_na = df[column].isna()
        if any(df_col_na):
            failed = True
            errors += f'\tfound {sum(df_col_na)} NaN values in column {column}.'
    if not failed:
        print_success(this_function_name)
    else:
        print_fail(this_function_name, errors)


def print_success(func_name: str):
    print(f'{func_name}: {bcolors.OKGREEN}SUCCESS!{bcolors.ENDC}')


def print_fail(func_name: str, reason: str = None):
    print(f'{func_name} {bcolors.FAIL}FAIL{bcolors.ENDC}' +
          f': {reason}' if reason is not None else '')


if __name__ == '__main__':
    main()
