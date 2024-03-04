import matplotlib.pyplot as plt
import numpy as np

from utils.makeDir import createDirectory

def vizMetrics(historys_loss, historys_sen, historys_spe, historys_bac, filename):
    for met in range(4):
        if met==0:
            metname='loss' 
            hs=historys_loss
        elif met==1:
            metname='sen'
            hs=historys_sen
        elif met==2:
            metname='spe'
            hs=historys_spe
        else: 
            metname='bac'
            hs=historys_bac

        sum_t=np.array(hs[0][f't_{metname}'])
        sum_tt=np.array(hs[0][f'tt_{metname}'])
        h={'epoch':hs[0]['epoch'], f'training_{metname}':[], f'test(val)_{metname}':[]}
        for i in range(1,10):
            sum_t += np.array(hs[i][f't_{metname}'])
            sum_tt += np.array(hs[i][f'tt_{metname}'])
        h[f'training_{metname}'] = sum_t/10
        h[f'test(val)_{metname}'] = sum_tt/10
        # print(f'avg training_{metname} : {h[f"training_{metname}"]}')
        # print(f'avg test_{metname} : {h[f"test(val)_{metname}"]}')
        
        plt.plot(h['epoch'], h[f'training_{metname}'], marker='.', c='blue', label = f'training_{metname}')
        plt.plot(h['epoch'], h[f'test(val)_{metname}'], marker='.', c='red', label = f'test(val)_{metname}')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel(f'avg {metname}')
        if metname=='loss':
            plt.ylim([0,10])
        else:
            plt.ylim([0,1])

        createDirectory(f'results/figs/new/{filename}')
        plt.savefig(f'results/figs/new/{filename}/{metname}.png')
        plt.show()







