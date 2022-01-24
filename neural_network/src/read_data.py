import pandas as pd

class read_data:

    def __init__(self):
        pass

    def get_dataframe(self):

        data = pd.read_csv('data/wdbc.data', sep=",")
        print(data)
