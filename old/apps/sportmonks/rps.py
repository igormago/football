import numpy as np
import pandas as pd


def testing():


    def rps_match(predictions, observed):

        sum1 = 0
        for i in range(0,2):
            sum2 = 0
            for j in range (0,i+1):
                calc = predictions[j] - observed[j]
                sum2 = sum2 + calc

            sum1 = sum1 + (sum2**2)

        return round(0.5*sum1,5)

    def rps_match2(predictions, observed):

        op1  = (predictions[0] - observed[0] + predictions[1] - observed[1])**2
        op2  = (predictions[0] - observed[0])**2

        return round(0.5*(op1+op2),5)


    columns = ['H','D','A']
    predictions = pd.DataFrame(columns=columns)
    observed = pd.DataFrame(columns=columns)
    predictions.loc[0] = [1.00,0.00,0.00]
    predictions.loc[1] = [0.90,0.10,0.00]
    predictions.loc[2] = [0.80,0.10,0.10]
    predictions.loc[3] = [0.50,0.25,0.25]
    predictions.loc[4] = [0.35,0.30,0.35]
    predictions.loc[5] = [0.60,0.30,0.10]
    predictions.loc[6] = [0.60,0.25,0.15]
    predictions.loc[7] = [0.60,0.15,0.25]
    predictions.loc[8] = [0.57,0.33,0.10]
    predictions.loc[9] = [0.60,0.20,0.20]

    observed.loc[0] = [1,0,0]
    observed.loc[1] = [1,0,0]
    observed.loc[2] = [1,0,0]
    observed.loc[3] = [1,0,0]
    observed.loc[4] = [0,1,0]
    observed.loc[5] = [0,1,0]
    observed.loc[6] = [1,0,0]
    observed.loc[7] = [1,0,0]
    observed.loc[8] = [1,0,0]
    observed.loc[9] = [1,0,0]


    for r in range(0,10):
        rps = rps_match2(predictions.iloc[r], observed.iloc[r])
        print(rps)
        rps = rps_match(predictions.iloc[r], observed.iloc[r])
        print(rps)


testing()
