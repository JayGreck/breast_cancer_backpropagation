
import pandas as pd
class Preprocess_Data:

    def __init__(self):
        pass

    def get_dataframe(self):

        

        df = pd.read_csv('data/wdbc.csv', sep=",")
        df = df.drop(df.columns[0], axis=1) # Dropping the first attribute (column)
        
        
        

        # ------------ Normalisation ------------ 

        # making a copy of the dataframe
        df_scaled1 = df.copy()

        # ------------ Label Encoding ------------ 
        df_scaled1['diagnosis'] = df_scaled1['diagnosis'].astype('category')
        df_scaled1['diagnosis'] = df_scaled1['diagnosis'].cat.codes
        
        df_scaled = df_scaled1.iloc[: , :-1]
        df_scaled = df_scaled.iloc[0: , :]
        
        
        for column_index in range(30):

            if column_index != 0: # Skipping target output
                # Normalising values between a range of 0 and 1
                df_scaled[df_scaled.columns[column_index]] = (df_scaled[df_scaled.columns[column_index]] 
                - df_scaled[df_scaled.columns[column_index]].min()) / (df_scaled[df_scaled.columns[column_index]].max() 
                - df_scaled[df_scaled.columns[column_index]].min())
        
        

        
        # ------------ Splitting Data ------------ 
         
        x_train70 = df_scaled.sample(frac = 0.7)
        dropped_classifier_X = x_train70.drop(x_train70.columns[0], axis=1) #dropping classifier 
        diagnosis_X = x_train70.loc[:, x_train70.columns.intersection(['diagnosis'])] # Gets the diagnosis 
        
        
       
        
        y_train1 = df_scaled.drop(x_train70.index) # Drops 70% and are now left with 30% for testing
        dropped_classifier_y = y_train1.drop(y_train1.columns[0], axis=1) #dropping classifier
        diagnosis_y = y_train1.loc[:, y_train1.columns.intersection(['diagnosis'])] # Gets the diagnosis 

        
        print("[X] Dropped First Attirbute \n[X] Normalised \n[X] 7:3 Split")
        
        X_train, X_test, y_train, y_test = dropped_classifier_X, diagnosis_X, dropped_classifier_y, diagnosis_y
        
        return X_train, X_test, y_train, y_test