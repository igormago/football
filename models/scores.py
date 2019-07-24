import pandas as pd
import numpy as np


def rps_error(y_true, y_hat_prob, **kwargs):

    y_true_prob = pd.DataFrame()
    y_true_prob['observed_home'] = (y_true == 'A') * 1
    y_true_prob['observed_draw'] = (y_true == 'D') * 1
    y_true_prob['observed_away'] = (y_true == 'H') * 1

    return rps(y_true_prob.values, y_hat_prob)


def rps(y_true_prob, y_hat_prob):

    op1 = np.square(y_hat_prob[:, 0] - y_true_prob[:, 0] + y_hat_prob[:, 1] - y_true_prob[:, 1])
    op2 = np.square(y_hat_prob[:, 0] - y_true_prob[:, 0])

    op3 = 0.5*(op1+op2)
    mean = np.mean(op3)

    return mean
