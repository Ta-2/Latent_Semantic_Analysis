import numpy as np
import illustrator as ill
import eigen_module as em

#データの取得
X = np.loadtxt('data.csv')
row = 5.0
col = 5.0

#取得データの確認
print("inputed data:")
print(X)

#データ行列とその転置した行列を掛け算する
XX = np.dot(X, X.T)

#Covの固有値と固有ベクトルを取得
eig_val, eig_vec = em.calc_sorter_eigen(XX)

print("eigen value:")
print(eig_val)
print("eigen vector:")
print(eig_vec)