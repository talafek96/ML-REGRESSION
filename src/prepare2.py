import numpy as np
from numpy.core.numeric import outer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import random
import re

# Same as prepare module, but merges blood_A and blood_AB to a single feature and drops the rest of the blood related features.


class DataImputer:
    def __init__(self, train_df, data_to_clean, features, strategy_dict, instructions_dict=dict(), missing_marker=np.NaN):
        self.strats = strategy_dict
        self.data_to_clean: pd.DataFrame = data_to_clean
        self.instructions_dict = instructions_dict
        self.train_df: pd.DataFrame = train_df
        self.missing_marker = missing_marker
        self.to_impute = features
        return

    def __median_impute(self, feature):
        median = self.train_df[feature].median()
        self.train_df[feature] = self.train_df[feature].fillna(median)
        self.data_to_clean[feature] = self.data_to_clean[feature].fillna(
            median)
        return

    def __random(self, feature):
        items = list(self.train_df[feature].value_counts().index)
        for i in range(len(self.train_df[feature])):
            if self.train_df.loc[i, feature] is self.missing_marker:
                self.train_df.loc[i, feature] = random.choice(items)

        for i in range(len(self.data_to_clean[feature])):
            if self.data_to_clean.loc[i, feature] is self.missing_marker:
                self.data_to_clean.loc[i, feature] = random.choice(items)
        return

    def __arbitrary(self, feature, constant):
        self.train_df[feature] = self.train_df[feature].fillna(constant)
        self.data_to_clean[feature] = self.data_to_clean[feature].fillna(
            constant)
        return

    def __frequent(self, feature):
        frequent_category = self.train_df[feature].value_counts().index[0]
        self.train_df[feature] = self.train_df[feature].fillna(
            frequent_category)
        self.data_to_clean[feature] = self.data_to_clean[feature].fillna(
            frequent_category)
        return

    def __bivariate_median(self, feature, secondary, init_bin_size=0):
        for i in range(self.train_df.shape[0]):
            if np.isnan(self.train_df.loc[i, feature]):
                tmp_secondary = self.train_df[secondary][i]
                tmp_bin_size = init_bin_size
                filtered_by_secondary = self.train_df[(self.train_df[secondary] >= tmp_secondary - tmp_bin_size) & (
                    self.train_df[secondary] <= tmp_secondary + tmp_bin_size)]
                while filtered_by_secondary[pd.notna(filtered_by_secondary[feature])].shape[0] == 0:
                    tmp_bin_size += 1
                    filtered_by_secondary = self.train_df[(self.train_df[secondary] >= tmp_secondary - tmp_bin_size) & (
                        self.train_df[secondary] <= tmp_secondary + tmp_bin_size)]
                bivariate_median = filtered_by_secondary[feature].median()
                self.train_df.loc[i, feature] = bivariate_median

        for i in range(self.data_to_clean.shape[0]):
            if np.isnan(self.data_to_clean.loc[i, feature]):
                tmp_secondary = self.train_df[secondary][i]
                tmp_bin_size = init_bin_size
                filtered_by_secondary = self.train_df[(self.train_df[secondary] >= tmp_secondary - tmp_bin_size) & (
                    self.train_df[secondary] <= tmp_secondary + tmp_bin_size)]
                while filtered_by_secondary[pd.notna(filtered_by_secondary[feature])].shape[0] == 0:
                    tmp_bin_size += 1
                    filtered_by_secondary = self.train_df[(self.train_df[secondary] >= tmp_secondary - tmp_bin_size) & (
                        self.train_df[secondary] <= tmp_secondary + tmp_bin_size)]
                bivariate_median = filtered_by_secondary[feature].median()
                self.data_to_clean.loc[i, feature] = bivariate_median
        return

    def impute_data(self):
        for feature in self.to_impute:
            if self.strats[feature] == 'median':
                self.__median_impute(feature)
            elif self.strats[feature] == 'random':
                self.__random(feature)
            elif self.strats[feature] == 'arbitrary':
                self.__arbitrary(feature, self.instructions_dict[feature])
            elif self.strats[feature] == 'frequent':
                self.__frequent(feature)
            elif self.strats[feature] == 'bivariate_median' or self.strats[feature] == 'bivariate':
                self.__bivariate_median(
                    feature, self.instructions_dict[feature][0], self.instructions_dict[feature][1])
        return self.train_df

    @staticmethod
    def impute_blood_ohe(train_df: pd.DataFrame, df_to_impute: pd.DataFrame) -> None:
        frequent_blood = train_df.blood_type.value_counts().index[0]
        regex = r'([ABO]{1,2})([+-])'
        match = re.search(regex, frequent_blood)
        blood_label = 'blood_' + match.group(1)
        rh_label = 'blood_' + match.group(2)
        for i in range(df_to_impute.shape[0]):
            if df_to_impute.loc[i, 'blood_nan'] == 1:
                df_to_impute.loc[i, blood_label] = 1
                df_to_impute.loc[i, rh_label] = 1
        df_to_impute.drop(columns='blood_nan', inplace=True)
        return


class OutlierCleaner:
    def __init__(self, df: pd.DataFrame, data_to_clean: pd.DataFrame):
        self.train_df = df
        self.data_to_clean = data_to_clean
        return

    def __z_score_clean(self, column, lowest, highest):
        if lowest is None:
            lowest = self.train_df[column].median() - 3 * \
                self.train_df[column].std()
        if highest is None:
            highest = self.train_df[column].median(
            ) + 3*self.train_df[column].std()
        self.train_df[column] = np.where(
            self.train_df[column] > highest,
            highest,
            np.where(
                self.train_df[column] < lowest,
                lowest,
                self.train_df[column]
            )
        )
        self.data_to_clean[column] = np.where(
            self.data_to_clean[column] > highest,
            highest,
            np.where(
                self.data_to_clean[column] < lowest,
                lowest,
                self.data_to_clean[column]
            )
        )

    def __iqr_clean(self, column, lowest, highest):
        percentile25 = self.train_df[column].quantile(0.25)
        percentile75 = self.train_df[column].quantile(0.75)
        iqr = percentile75 - percentile25
        if lowest is None:
            lowest = percentile25 - 1.5*iqr
        if highest is None:
            highest = percentile75 + 1.5*iqr
        self.train_df[column] = np.where(
            self.train_df[column] > highest,
            highest,
            np.where(
                self.train_df[column] < lowest,
                lowest,
                self.train_df[column]
            )
        )
        self.data_to_clean[column] = np.where(
            self.data_to_clean[column] > highest,
            highest,
            np.where(
                self.data_to_clean[column] < lowest,
                lowest,
                self.data_to_clean[column]
            )
        )

    # Consider making this a @staticmethod func and pass in a df to filter.
    def z_score_filter(self, column):
        highest = self.train_df[column].mean() + 3*self.train_df[column].std()
        lowest = self.train_df[column].mean() - 3*self.train_df[column].std()
        return self.train_df[(self.train_df[column] > highest) | (self.train_df[column] < lowest)]

    # Consider making this a @staticmethod func and pass in a df to filter.
    def iqr_filter(self, column):
        percentile25 = self.train_df[column].quantile(0.25)
        percentile75 = self.train_df[column].quantile(0.75)
        iqr = percentile75 - percentile25
        highest = percentile75 + 1.5*iqr
        lowest = percentile25 - 1.5*iqr
        return self.train_df[(self.train_df[column] > highest) | (self.train_df[column] < lowest)]

    def clean_outliers(self, column, filter, lowest=None, highest=None):
        '''
        Cleans the outliers of the column of the dataframe managed in the class *in-place*.

                Parameters:
                ----------
                        column (str): The feature's column to be cleaned from outliers.
                        filter (str): The cleaning method. Either 'z-score' or 'iqr'.
                        lowest (Optional[int]): If the filter 'iqr' was chosen and lowest was stated, the bottom bound will be set to this value.
                        highest (Optional[int]): If the filter 'iqr' was chosen and highest was stated, the upper bound will be set to this value.

                Returns:
                -------
                        The dataframe managed in the class (the original, to allow pipelining).
        '''
        if filter.lower() == 'z_score':
            self.__z_score_clean(column, lowest, highest)
        elif filter.lower() == 'iqr':
            self.__iqr_clean(column, lowest, highest)
        else:
            raise Exception("Method not supported.")
        return self.train_df


class SymptomExtracter:
    def __init__(self, df) -> None:
        self.df = df
        self.symptom_set = set()
        for symptoms in self.df.symptoms:
            if symptoms is np.NaN:
                continue
            symptoms = symptoms.split(';')
            self.symptom_set = self.symptom_set.union(set(symptoms))
        self.symptom_dict = dict(
            zip(self.symptom_set, range(len(self.symptom_set))))

    def getSymptomList(self, entry_index):
        '''
        Returns the symptom list that the patient at index <entry_index> has in the <df> dataframe.
        Assumes the existence of the column 'symptoms' that has a list of features seperated by a ';' character.
        '''
        symptoms = self.df.loc[entry_index, 'symptoms']
        if symptoms is np.NaN:
            return [np.NaN]*len(self.symptom_set)
        res = [0]*len(self.symptom_set)
        symptoms = symptoms.split(';')
        for symptom in symptoms:
            res[self.symptom_dict[symptom]] = 1
        return res


class data_extraction:
    def __init__(self, df) -> None:
        self.data: pd.DataFrame = df
        return

    # TODO: Realign the columns
    def extract_symptoms(self):
        sm = SymptomExtracter(self.data)
        symptoms_df = pd.DataFrame([sm.getSymptomList(i) for i in range(
            self.data.shape[0])], columns=list(sm.symptom_set))
        data = self.data.join(symptoms_df)
        data.drop(labels=["symptoms"], axis=1, inplace=True)
        cols = data.columns.tolist()
        cols = cols[:11] + symptoms_df.columns.tolist() + \
            cols[11:len(cols) - symptoms_df.shape[1]]
        self.data = data[cols]
        return

    def extract_blood_ohe(self):
        blood_types = list()
        rh_enzymes = list()
        trans_dict = {"A+": "A", "A-": "A",
                      "B+": "B", "B-": "B",
                      "AB+": "AB", "AB-": "AB",
                      "O+": "O", "O-": "O"}
        for blood in self.data.blood_type:
            blood_types.append(blood if blood is np.NaN else trans_dict[blood])
            rh_enzymes.append(blood if blood is np.NaN else blood[-1])

        blood_ohe = pd.get_dummies(
            pd.Series(blood_types), dummy_na=False, prefix="blood")
        rh_ohe = pd.get_dummies(pd.Series(rh_enzymes),
                                dummy_na=True, prefix="blood")
        full_blood_ohe: pd.DataFrame = pd.concat([blood_ohe, rh_ohe], axis=1)
        for column in full_blood_ohe.columns[::-1]:
            self.data.insert(5, column=column, value=full_blood_ohe[column])
        return

    def extract_zip_code(self):
        zip_list = list()
        for address in self.data.address:
            if address is not np.NaN:
                match = re.search(r'[A-Z]{2}\s+(\d+)$', address)
                zip_list.append(int(match.group(1)))
            else:
                zip_list.append(np.NaN)
        self.data.insert(5, "zip_code", zip_list)
        self.data.drop(labels='address', inplace=True, axis=1)
        return

    def apply_all(self):
        self.extract_symptoms()
        self.extract_blood_ohe()
        self.extract_zip_code()
        return self.data


def date_to_num(date: str) -> int:
    regex = r'(\d{4})-(\d{2})-(\d{2})'
    match = re.search(regex, date)
    return int(match.group(1) + match.group(2) + match.group(3))


def imput_data(data_to_clean, train):
    FEATURES_TO_IMPUTE = ['weight', 'age', 'sex', 'zip_code',
                          'num_of_siblings', 'happiness_score', 'household_income',
                          'fever', 'low_appetite', 'shortness_of_breath', 'cough',
                          'headache', 'conversations_per_day', 'PCR_10',
                          'sugar_levels', 'sport_activity', 'PCR_01', 'PCR_02',
                          'PCR_03', 'PCR_04', 'PCR_06', 'PCR_05', 'PCR_07',
                          'PCR_08', 'PCR_09']
    strat_dict = {
        'age': 'bivariate_median',  # with weight
        'sex': 'random',
        'weight': 'median',
        'zip_code': 'median',
        # 'x_location': 'median',
        # 'y_location': 'median',
        'num_of_siblings': 'median',
        'happiness_score': 'median',
        'household_income': 'bivariate_median',  # with age
        'fever': 'arbitrary',
        'low_appetite': 'arbitrary',
        'shortness_of_breath': 'arbitrary',
        'cough': 'arbitrary',
        'headache': 'arbitrary',
        # 'pcr_date': 'median',
        'conversations_per_day': 'median',
        'sugar_levels': 'bivariate_median',  # with weight
        'sport_activity': 'median',
        'PCR_01': 'median',
        'PCR_02': 'bivariate_median',  # with PCR_01
        'PCR_03': 'median',
        'PCR_04': 'bivariate_median',  # with PCR_03
        'PCR_05': 'bivariate_median',  # with PCR_06
        'PCR_06': 'bivariate_median',  # with PCR_10
        'PCR_07': 'bivariate_median',  # with PCR_10
        'PCR_08': 'bivariate_median',  # with PCR_04
        'PCR_09': 'bivariate_median',  # with PCR_10
        'PCR_10': 'median',
    }
    instructions = {
        'age': ('weight', 5),
        'household_income': ('age', 2),
        'fever': 0,
        'low_appetite': 0,
        'shortness_of_breath': 0,
        'cough': 0,
        'headache': 0,
        'sugar_levels': ('weight', 5),
        'PCR_02': ('PCR_01', 0.05),
        'PCR_04': ('PCR_03', 1),
        'PCR_05': ('PCR_06', 1),
        'PCR_06': ('PCR_10', 0.5),
        'PCR_07': ('PCR_10', 0.5),
        'PCR_08': ('PCR_04', 15),
        'PCR_09': ('PCR_08', 1),
    }
    di = DataImputer(train, data_to_clean, FEATURES_TO_IMPUTE,
                     strat_dict, instructions)
    di.impute_data()
    DataImputer.impute_blood_ohe(train, data_to_clean)
    return


def mergeBloodA(df: pd.DataFrame):
    df.insert(5, column='contains_blood_A', value=(
        (df.blood_A == 1) | (df.blood_AB == 1)).astype(int))


def prepare_data(data, training_data, drop_id=True):
    '''
    Returns a cleaned copy of data ready to be used for prediction.

            Parameters:
                    data (pandas.DataFrame): The dataframe to be cleaned.
                    training_data (pandas.DataFrame): The training set dataframe used to clean according to.

            Returns:
                    clean_data (pandas.DataFrame): A *copy* of data after it has been cleaned relatively to the provided training_data.
    '''
    # Copy the input dataframes:
    data_copy = data.copy()
    train_copy = training_data.copy()

    # Extract the new features:
    train_ex = data_extraction(train_copy)
    data_ex = data_extraction(data_copy)
    train_copy = train_ex.apply_all()
    data_copy = data_ex.apply_all()

    # Clean outliers:
    outlier_cleaner = OutlierCleaner(train_copy, data_copy)
    z_score_set = {'sugar_levels', 'PCR_01',
                   'PCR_02', 'PCR_06', 'PCR_07'}
#   iqr_set = {'household_income', 'num_of_siblings', 'PCR_03', 'PCR_05', 'PCR_10'}
    features_to_clean = ['household_income', 'num_of_siblings', 'sugar_levels', 'PCR_01',
                         'PCR_02', 'PCR_03', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_10']
    for feature in features_to_clean:
        outlier_cleaner.clean_outliers(
            feature, 'z_score' if feature in z_score_set else 'iqr')

    # Impute missing data:
    imput_data(data_copy, train_copy)

    # Perform feature selection:
    mergeBloodA(data_copy)
    cols_to_keep = ['zip_code', 'contains_blood_A', 'household_income',
                    'num_of_siblings', 'shortness_of_breath', 'fever', 'sugar_levels',
                    'PCR_01', 'PCR_02', 'PCR_03', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_10',
                    'VirusScore']
    if not drop_id:
        cols_to_keep.insert(0, 'patient_id')
    
    data_copy = data_copy[cols_to_keep]

    pd.options.mode.chained_assignment = None  # HACK
    # Perform feature normalization:
    minmax = MinMaxScaler()
    standardizator = StandardScaler()
    features_to_norm = {
        ('zip_code', minmax), ('household_income',
                               minmax), ('num_of_siblings', minmax),
        ('sugar_levels', standardizator), ('PCR_01',
                                           standardizator), ('PCR_02', standardizator),
        ('PCR_03', standardizator), ('PCR_05',
                                     standardizator), ('PCR_06', standardizator),
        ('PCR_07', standardizator), ('PCR_10', standardizator)
    }

    for feature in features_to_norm:
        col_data = np.array(data_copy[feature[0]]).reshape(-1, 1)
        col_train = np.array(train_copy[feature[0]]).reshape(-1, 1)
        feature[1].fit(col_train)
        data_copy[feature[0]] = feature[1].transform(col_data)
        train_copy[feature[0]] = feature[1].transform(col_train)

    pd.options.mode.chained_assignment = 'warn'  # HACK
    return data_copy
