import pandas as pd
class Preprocess_Data:

    def __init__(self):
        pass

    def get_dataframe(self):

        df = pd.read_csv('data/wdbc.data', sep=",")
        df = df.drop(df.columns[0], axis=1) # Dropping the first attribute (column)

        # ------------ Normalisation ------------ 

        # making a copy of the dataframe
        df_scaled = df.copy()

        # ------------ Label Encoding ------------ 
        df_scaled['M'] = df_scaled['M'].astype('category')
        df_scaled['M'] = df_scaled['M'].cat.codes
        

        for column_index in range(31):

            if column_index != 0: # Skipping target output
                # Normalising values between a range of 0 and 1
                df_scaled[df_scaled.columns[column_index]] = (df_scaled[df_scaled.columns[column_index]] 
                - df_scaled[df_scaled.columns[column_index]].min()) / (df_scaled[df_scaled.columns[column_index]].max() 
                - df_scaled[df_scaled.columns[column_index]].min())
        
        
        # ------------ Splitting Data ------------ 
        x_train = df_scaled.sample(frac = 0.7) 
        y_test = df_scaled.drop(x_train.index) # Drops 70% and are now left with 30% for testing
        print("[X] Dropped First Attirbute \n[X] Normalised \n[X] 7:3 Split")
        
        return x_train, y_test