import numpy as np

metname='sen'
hs =[{'epoch': [0, 1], 't_sen': [0.5178571428571429, 0.3382352941176471], 'tt_sen': [0.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.58, 0.1774193548387097], 'tt_sen': [0.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.5964912280701754, 0.45098039215686275], 'tt_sen': [0.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.6, 0.2222222222222222], 'tt_sen': [0.4, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.5714285714285714, 0.46875], 'tt_sen': [0.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.5454545454545454, 0.7096774193548387], 'tt_sen': [1.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.5862068965517241, 1.0], 'tt_sen': [1.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.5151515151515151, 0.48333333333333334], 'tt_sen': [0.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.5079365079365079, 0.0], 'tt_sen': [0.0, 1.0]}, 
    {'epoch': [0, 1], 't_sen': [0.5333333333333333, 0.0], 'tt_sen': [0.0, 1.0]}]

sum_t=np.array(hs[0][f't_{metname}'])
sum_tt=np.array(hs[0][f'tt_{metname}'])
print(sum_tt)
for i in range(1,10):
    sum_t += np.array(hs[i][f't_{metname}'])
    sum_tt += np.array(hs[i][f'tt_{metname}'])
    print(sum_tt)
print(f'training_{metname} : {sum_t/10}')
print(f'test_{metname} : {sum_tt/10}')