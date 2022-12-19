from typing import Union
import pandas as pd
import numpy as np

def findCorrelation(df: pd.DataFrame, threshold: Union[float,int], exact: bool = True, names: bool = False) -> list:
    """ Calculates the 'df' correlation matrix and returns a vector with columns to remove to reduce pair-wise (absolute) correlations
    Details: if the correlation between two variables is greater than the cutoff, eliminates the one with the higher mean (absolute) correlation considering ALL variables
    
    Args:
        df (pd.DataFrame): a matrix of data
        threshold (float, int): A value between [0,1] to use as the cutoff for pair-wise correlation elimination

        exact (bool, optional): If exact==True, recomputes the mean (absolute) correlation of each variable every time there is an elimination; 
                                If exact==False, computes the mean (absolute) correlation of each variable only with the initial corr matrix. Defaults to True.

        names (bool, optional): If names==True, returns the labels to remove from the original df; 
                                if names==False, returns integers representing the indices to remove from the original df. Defaults to False.

    Returns:
        (list): a list containing the columns to remove from 'df'

    Obs.: this is a Python equivalent for the findCorrelation function from the R language. 
    Different from the R version, this function takes a dataframe and calculates a correlation matrix (instead of taking a correlation matrix as input)
    https://www.rdocumentation.org/packages/caret/versions/6.0-93/topics/findCorrelation
     
    """
    corr_matrix = abs(np.array(df.corr())) #computes the (absolute) correlation matrix ignoring NaN
    np.fill_diagonal(corr_matrix,val=np.nan) #fills the main diagonal with NaN to facilitate posterior logic

    if not exact: #if exact=False, computes the individuals mean absolute correlations just one time 
        sums = np.nanmean(corr_matrix, axis=1) # mean absolute correlations for each variable (actually using sum to save one (the division by "n") operation)

    loop = True # control flow variable; while loop = True, continues eliminating variables
    remove = [] # placeholder list for the columns to remove from the original df

    while loop:
        # main logic: first line returns a tuple with (line,column) position of the highest pairwise correlation exceeding the "threshold";
            # correlation matrix is updated with NaNs in the entire line/column corresponding to the eliminated variable
            # if no correlation exceeds the threshold, tuple equals to (0,0) and the loop is terminated
                # @obs: corr_matrix>threshold will give FALSE for NaN's; argmax will always return the first "True/False"; hence, when the matrix has only corr < threshold OR NaNs, the line will return (0,0)
        (idx1,idx2) = np.unravel_index(np.argmax(corr_matrix>threshold), corr_matrix.shape) 

        if idx1!=idx2:          #if there is any corr>threshold, eliminates one variable
            
            if exact:           # if exact == true, recomputes candidate variables mean (absolute) correlation before comparison
                mean_corr1 = np.nanmean(corr_matrix[:,idx1])
                mean_corr2 = np.nanmean(corr_matrix[:,idx2])

                np.delete(corr_matrix[:,2],2)

            else: # if exact == false, use the candidate variables mean (absolute) correlation of the initial corr matrix
                mean_corr1, mean_corr2 = sums[idx1], sums[idx2]

            if mean_corr1 > mean_corr2:
                remove.append(idx1)             # stores the index position of the variable in the ORIGINAL df 

                # Below: "eliminates" the variable from the correlation matrix by placing NaNs in the entire row/columns
                    # @obs: placing NaNs is better than resizing the matrix by using np.del, as this preserves index positions of the ORIGINAL "df" - which is what we want
                corr_matrix[:,idx1]=np.nan 
                corr_matrix[idx1,:]=np.nan        
            else:
                remove.append(idx2)
                corr_matrix[:,idx2]=np.nan
                corr_matrix[idx2,:]=np.nan
        else:
            loop = False

    remove = list(df.columns[remove]) if names else remove      #return df column labels or column index positions?
    return remove