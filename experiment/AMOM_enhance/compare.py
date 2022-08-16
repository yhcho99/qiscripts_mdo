import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import os
import statistics
from pandas import DataFrame

storage_path = "/raid/sr-storage/"

coef_ls_param = [
    {
        'm12c':1.0,
        'm6c':1.0,
        'm3c':1.0
    },{
        'm12c':1.0,
        'm6c':1.0,
        'm3c':0.0
    },{
        'm12c':1.0,
        'm6c':0.0,
        'm3c':1.0
    },{
        'm12c':0.0,
        'm6c':1.0,
        'm3c':1.0
    }
]

num_exp = 3 # number of experiments
num_case = len(coef_ls_param)
feat_names = ['Total Return', 'Total Sharpe','Total Mdd']
num_feat = len(feat_names) # return, sharpe, mdd

def gather_info(IDENTIFIER, coef_ind, test_id, storage_path):
    '''
    Collect all stats.csv files from sr-storage and extract chosen information 
    '''
    total_ret_name = IDENTIFIER+'_total_return'
    total_shp_name = IDENTIFIER+'_total_sharpe'
    total_mdd_name = IDENTIFIER+'_total_mdd'

    if IDENTIFIER in os.listdir(storage_path):
        df = pd.read_csv((os.path.join(storage_path, IDENTIFIER + '/comparison/stats.csv')))
        df.rename(columns = {'Unnamed: 0':'date', total_ret_name:'total_return', total_shp_name:"total_sharpe", total_mdd_name:"total_mdd"}, inplace = True)
        df['momentum coef'] = str(coef_ind['m12c']) + "  " + str(coef_ind['m6c']) +"  " + str(coef_ind['m3c'])
        df['exp_num'] = test_id
        return df[['momentum coef','total_return', 'total_sharpe', 'total_mdd','exp_num']]
    
def center_align(df: DataFrame):
    '''
    Enforce cell center aligned
    '''
    aligned_df = df.style.set_properties(**{'text-align': 'center'})
    aligned_df = aligned_df.set_table_styles(
        [dict(selector='th', props=[('text-align', 'center')])]
    )
    return aligned_df

if __name__ == "__main__":
    print ("**** Gather Results ****")
    total_result = []
    for coef_ind in coef_ls_param:
        for test_ind in range(1, num_exp+1):
            m12_coef = coef_ind['m12c']
            m6_coef = coef_ind['m6c']
            m3_coef = coef_ind['m3c']
            test_id = str(test_ind) # string number
            IDENTIFIER = f'amom_enhance_test_vol_adj_m12x{m12_coef}_m6x{m6_coef}_m3x{m3_coef}+testNum{test_id}'
            
            result = gather_info(IDENTIFIER, coef_ind, test_ind, storage_path)
            total_result.append(result)
    all = pd.concat(total_result, axis = 0)

    # Data Preprocessing 
    mom_coef_ls = list(all['momentum coef'].unique()) # -> (1,1,0), (1,1,1,), ...
    exp_ind = [i for i in range(1, num_exp+1)] # -> experiment 1, experiment 2, experiment 3
    opt_all = [[] for _ in range(len(exp_ind))]

    for i, exp_id in enumerate(exp_ind):
        for seq_ind in (range(num_case)):    
            d3 = all[all['exp_num'] == exp_id][['total_return', 'total_sharpe','total_mdd']].T.iloc[:,seq_ind]
            opt_all[i].append(d3)
            
        opt_all[i] = pd.concat(opt_all[i], axis = 0)
    opt_all = pd.concat(opt_all, axis = 1)
    total_arr = opt_all.values
    
    # Multiindex for columns 
    index = pd.MultiIndex.from_product([mom_coef_ls, feat_names],
                                    names=['m12 m6  m3', 'Features'])
    columns = pd.MultiIndex.from_product([['Experiment'], [str(i) for i in range(1, num_exp+1)]]) 
    final_df = pd.DataFrame(total_arr, index=index, columns=columns)

    # Center alignment
    final_df2 = center_align(final_df)
    
    # Saving the result
    print ("**** Saving Done ****")
    final_df2.to_excel('./Compare_Result.xlsx') # Styler only goes along with excel type 
