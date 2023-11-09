import numpy as np

def calc_sorter_eigen(matrix):
    #固有値、固有ベクトルの計算
    eig_val,  eig_vec = np.linalg.eig(matrix)
    eig_vec = eig_vec.T

    #固有値が全て正の値になるように調節
    for idx, val in enumerate(eig_val):
        if val < 0.0:
            eig_val[idx] *= -1
            eig_vec[idx] *= -1

    #固有値の値が降順になるように固有値ベクトルと固有ベクトル行列をソート
    eig_set = zip(eig_val, eig_vec)
    sorted_eig_set = sorted(eig_set, key=lambda x: x[0], reverse=True)
    eig_val = np.array( [se[0] for se in sorted_eig_set] )
    eig_vec = np.array( [se[1] for se in sorted_eig_set] )

    return eig_val, eig_vec