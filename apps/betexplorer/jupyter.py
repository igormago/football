import copy

from IPython.core.display import display
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
import pandas as pd
from modules.betexplorer2.notations import DataFrameNotation, MatchNotation

CHAMPIONSHIP_NAME = DataFrameNotation.from_matches(MatchNotation.championship_name())
YEAR = DataFrameNotation.from_matches(MatchNotation.season_initial_year())

BOOK_HIT = 'm.hit.bts.f'
ODDS_YES = 'm.odds.s.bts.y.o.avg'
ODDS_NO = 'm.odds.s.bts.n.o.avg'
STD_YES = 'm.odds.s.bts.n.o.std'
STD_NO = 'm.odds.s.bts.n.o.std'
FAVORITE_BTS = 'm.odds.f.bts.f.c'
UNDERDOG_BTS = 'm.odds.f.bts.u.c'

DADOS = 'Dados'
JOGOS = 'Jogos'
TOTAL = 'Total'

P_VALUE = 'p-value'
T_TEST = 'T-Test'
T_STATISTIC = 't-statistic'

YES = 'y'
NO = 'n'

ACERTOU='Acertou'
ERROU='Errou'
ACERTOU_ERROU = [ACERTOU,ERROU]

LINHAS= 'Linhas'
COLUNAS= 'Colunas'
LINHAS_COLUNAS = [LINHAS, COLUNAS]

SIM='Sim'
NAO='Não'
SIM_NAO = [SIM,NAO]


class Resume:

    def __init__(self, df):

        self.resume = pd.DataFrame()
        self.df = df

        self.resume = self.df.groupby([CHAMPIONSHIP_NAME]).size()
        self.resume = self.resume.to_frame()
        self.resume.columns = [TOTAL]

        self.total = pd.DataFrame()

        self.championships = self.df[CHAMPIONSHIP_NAME].unique()
        self.columns = list()

    def get_resume(self):
        return self.resume

    def add_count_matches_by_year(self):

        df = self.df
        resume_by_year = df.groupby([CHAMPIONSHIP_NAME, YEAR]).size()
        resume_by_year = resume_by_year.to_frame()
        resume_by_year = resume_by_year.reset_index()
        resume_by_year = resume_by_year.pivot(index=CHAMPIONSHIP_NAME, columns=YEAR, values=0)
        self.resume = pd.concat([self.resume, resume_by_year], axis=1)

    def add_total(self):
        self.resume.loc[TOTAL] = self.resume.sum()

    def add_comparing(self, df=None, label=None, column=None, other_column=None, result=None):

        if df is None:
            df = self.df

        if (other_column is None and result is None) or (other_column is not None and result is not None):
            raise Exception('Problema')
        else:
            if other_column:
                group_equals = df[df[column] == df[other_column]].groupby([CHAMPIONSHIP_NAME]).size()
            else:
                group_equals = df[df[column] == result].groupby([CHAMPIONSHIP_NAME]).size()

            self.resume[label] = group_equals

        self.columns.append(str(label))

    def add_test_1samp(self, column):

        df = self.df
        for c in self.championships:
            dist = df[df[CHAMPIONSHIP_NAME] == c][column]
            res_test = ttest_1samp(dist, 0.5)
            self.resume.loc[c, T_STATISTIC] = round((res_test[0]), 5)
            self.resume.loc[c, P_VALUE] = round((res_test[1]), 5)

        dist = df[column]
        res_test = ttest_1samp(dist, 0.5)
        self.resume.loc[TOTAL, T_STATISTIC] = round((res_test[0]), 5)
        self.resume.loc[TOTAL, P_VALUE] = round((res_test[1]), 5)

    def add_test_2samp(self, df1=None, df2=None, column1=None, column2=None, ind=True):

        if df1 is None:
            df1 = self.df

        if df2 is None:
            df2 = self.df

        for c in self.championships:
            df_dist1 = df1[df1[CHAMPIONSHIP_NAME] == c]
            df_dist2 = df2[df2[CHAMPIONSHIP_NAME] == c]
            dist1 = df_dist1[column1]
            dist2 = df_dist2[column2]
            if ind:
                res_test = ttest_ind(dist1, dist2)
            else:
                res_test = ttest_rel(dist1, dist2)
            self.resume.loc[c, T_STATISTIC] = round((res_test[0]), 5)
            self.resume.loc[c, P_VALUE] = round((res_test[1]), 5)

        dist1 = df1[column1]
        dist2 = df2[column2]
        if ind:
            res_test = ttest_ind(dist1, dist2)
        else:
            res_test = ttest_rel(dist1, dist2)
        self.resume.loc[TOTAL, T_STATISTIC] = round((res_test[0]), 5)
        self.resume.loc[TOTAL, P_VALUE] = round((res_test[1]), 5)

    def add_comparing_sum(self, label, column, column2=None, result=None, colum_sum=None):

        df = self.df
        resume = self.resume
        if column2 is None:
            total = df[df[column] == result].groupby(CHAMPIONSHIP_NAME)[colum_sum].sum()
        else:
            total = df[df[column] == df[column2]].groupby(CHAMPIONSHIP_NAME)[colum_sum].sum()

        resume[label] = round(total, 2)
        self.columns.append(label)

    def add_profit_loss(self):
        new_cols = list()
        for c in self.columns:
            label = c + '_pl'
            self.resume[label] = self.resume[c] - self.resume[TOTAL]
            new_cols.append(label)

        self.columns.extend(new_cols)

    def display(self, percentage=False, sort_by=None):

        def highlight_total_row(s):
            test = s.index == TOTAL
            return ['background-color: #DDDDDD; font-weight: bold' if v else '' for v in test]

        def highlight_total_column(s):
            return ['background-color: #DDDDDD; font-weight: bold' if v else '' for v in s]

        def highlight_p_value(s):
            test = s > 0.05
            return ['color: #c0392b;' if v else 'color: #16a085' for v in test]

        def highlight_lg_zero(s):
            test = s < 0
            return ['color: #c0392b' if v else 'color: #16a085' for v in test]

        resume = copy.copy(self.resume)

        if percentage:
            for c in self.columns:
                resume[c] = round(resume[c] / resume[TOTAL] * 100, 2)

        if sort_by:
            resume.sort_values(by=sort_by, inplace=True)

        resume = resume.style.apply(highlight_total_row)
        resume.apply(highlight_total_column, subset=[TOTAL])

        if P_VALUE in (resume.columns):
            resume.apply(highlight_p_value, subset=[P_VALUE])

        cols_PL = [col for col in self.columns if col.endswith('pl') or col.endswith('%')]

        if len(cols_PL) > 0:
            resume.apply(highlight_lg_zero, subset=cols_PL)

        display(resume)
