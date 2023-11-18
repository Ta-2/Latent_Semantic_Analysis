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
XX = np.dot(X.T, X)
print("XX:")
print(XX)

#Covの固有値と固有ベクトルを取得
eig_val, eig_vec = em.calc_sorter_eigen(XX)

print("eigen value:")
print(eig_val)
print("eigen vector:")
print(eig_vec)
eig_val_s = np.sqrt(eig_val)

#ここから特異値分解（X = UΣV）の各行列を求める
#行列Σを求める
Sigma = np.diag(eig_val_s)
print("Sigma:")
print(Sigma)

#行列Vを求める
norm = np.linalg.norm(eig_vec, axis=1)
V = (eig_vec.T/norm).T
print("V:")
print(V)

#行列Uを求める
eig_val_inv = np.where(eig_val>0.0, 1/eig_val_s, 0.0)
#eig_val_inv = 1.0/eig_val_s
Sigma_inv = np.diag(eig_val_inv)
print("Sigma_inv:")
print(Sigma_inv)
V_inv = V.T
U = np.dot(X, np.dot(V_inv, Sigma_inv))
print("U:")
print(U)

A = np.dot(U, np.dot(Sigma, V))
print("Check UΣV:")
print(A)

#寄与率、累積寄与率に相当する数字を計算
Proportion_of_Variance = eig_val_s/np.sum(eig_val_s)
Cumulative_Proportion = []
CPsum = 0.0
for PoV in Proportion_of_Variance:
    Cumulative_Proportion.append(PoV+CPsum)
    CPsum = PoV+CPsum
    
print("Proportion of Variance: ")
print(Proportion_of_Variance)
print("Cumulative Proportion: ")
print(Cumulative_Proportion)

#新しい基底での座標を計算する
basis_num = 2   #用意する基底の数
print(V[0:2,:])
dimension_pressed_data = np.dot(X, V[0:2,:].T)
print("dimension_pressed_data:")
print(dimension_pressed_data)

ill.illustrate(dimension_pressed_data)
ill.show()