# -*- coding: utf-8 -*-
# @Author: Konstantin Schuckmann
# @Date:   2021-10-28 14:36:06
# @Last Modified by:   Konstantin Schuckmann
# @Last Modified time: 2021-10-29 09:30:48

import pandas as pd
import numpy as np

# Needed for generating data from an existing dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer



def create_dummy_variables_from_original():
    data = pd.read_csv('./data/input/input.csv', sep=';')
    column_names = ['Distance','Rate 0 - 49 kg', 'Rate 50 - 99 kg', 'Rate 99 - 149 kg',
       'Rate 149 - 199 kg', 'Rate 199 - 299 kg', 'Rate 299 - 399 kg',
       'Rate 399 - 499 kg', 'Rate 499 - 599 kg', 'Rate 599 - 699 kg',
       'Rate 699 - 799 kg', 'Rate 799 - 899 kg', 'Rate 899 - 999 kg',
       'Rate 999 - 1099 kg', 'Rate 1099 - 1199 kg', 'Rate 1199 - 1299 kg',
       'Rate 1299 - 1399 kg', 'Rate 1399 - 1499 kg', 'Rate 1499 - 1749 kg',
       'Rate 1749 - 1999 kg', 'Rate 1999 - 2249 kg', 'Rate 2249 - 2499 kg',
       'Rate 2499 - 2999 kg', 'Rate 2999 - 3499 kg', 'Rate 3499 - 3999 kg',
       'Rate 3999 - 4499 kg', 'Rate 4499 - 4999 kg', 'Rate 4999 - 5999 kg',
       'Rate 5999 - 6999 kg', 'Rate 6999 - 7999 kg', 'Rate 7999 - 8999 kg',
       'Rate 8999 - 9999 kg', 'Rate 9999 - 10999 kg', 'Rate 10999 - 11999 kg',
       'Rate 11999 - 12999 kg', 'Rate 12999 - 13999 kg',
       'Rate 13999 - 14999 kg', 'Rate 14999 - 15999 kg',
       'Rate 15999 - 16999 kg', 'Rate 16999 - 17999 kg',
       'Rate 17999 - 18999 kg', 'Rate 18999 - 19999 kg', 'Rate 20000-21499 kg',
       'Rate 21500-22999 kg', 'Rate 23000-24499 kg']
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(data[['GDP WB origin', 'GDP WB Destination',
       'Lat Origin', 'Lon Origin', 'Lat Destination', 'Lon Destination']])
    data[['GDP WB origin', 'GDP WB Destination',
       'Lat Origin', 'Lon Origin', 'Lat Destination', 'Lon Destination']] = imp.transform(data[['GDP WB origin', 'GDP WB Destination',
       'Lat Origin', 'Lon Origin', 'Lat Destination', 'Lon Destination']])
    
    sorted_data = data.sort_values(by=['Distance'])
    rand_state = 11
    
    # Fetch the dataset and store in X
    X = data[column_names]
    
    # Fit a kernel density model using GridSearchCV to determine the best parameter for bandwidth
    bandwidth_params = {'bandwidth': np.arange(0.01,1,0.05)}
    grid_search = GridSearchCV(KernelDensity(), bandwidth_params)
    grid_search.fit(X)
    kde = grid_search.best_estimator_
    
    # Generate/sample 
    new_data = kde.sample(sorted_data.shape[0], random_state=rand_state)

    new_data = pd.DataFrame(new_data, columns = column_names)
    final_df = pd.concat([data[['Country Relation', 'Country Relation Vice Versa', 'Origin Country',
       'Destination Country', 'GDP WB origin', 'GDP WB Destination',
       'Lat Origin', 'Lon Origin', 'Lat Destination', 'Lon Destination']], new_data], axis = 1)
   
    final_df.to_csv('./data/input/dummy_input.csv', sep=';', index=False)
    
def main():
    create_dummy_variables_from_original()

if __name__ == '__main__':
    main()