## plot the convergence curve of each experiment

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# fig, ax = plt.subplots()
# ax.plot(100*np.random.rand(20))

# formatter = ticker.FormatStrFormatter('$%1.2f')
# ax.yaxis.set_major_formatter(formatter)

# for tick in ax.yaxis.get_major_ticks():
#     tick.label1On = False
#     tick.label2On = True
#     tick.label2.set_color('green')

# plt.show()

import json 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# zhfont1 = matplotlib.font_manager.FontProperties(fname='/Users/morino/Library/Fonts/msyh.ttf')

ml_qsvd = {
	"RMSE": [
	1.20653083023 
, 0.914448996814
, 0.9040892614
, 0.901226118938
, 0.898096467204
, 0.898982392509
, 0.898352472199
, 0.899226199334
, 0.898727687511
, 0.899010172446
, 0.899654520483
, 0.899607929873
, 0.900822904975
, 0.901051215102
, 0.903215462665
, 0.902816463368
, 0.902787091778
, 0.90362216325 
, 0.903550728215
, 0.904100218222
, 0.898096467204],
	"MAE": [
1.00442130127
, 0.710547875977
, 0.69955614624
, 0.696783413696
, 0.693013925171
, 0.694998391724
, 0.692470269775
, 0.694106530762
, 0.693112329102
, 0.693299960327
, 0.69380871582
, 0.69352010498
, 0.694082797241
, 0.69393187561
, 0.695816799927
, 0.695083685303
, 0.695876461792
, 0.696061825562
, 0.695659893799
, 0.695876077271
, 0.692470269775
	]	
}

bc_qsvd = {
	"RMSE" : [
	2.44704094436
, 1.78705058776
, 1.73363005337
, 1.71327230012
, 1.70332808054
, 1.69879065506
, 1.69636021506
, 1.69584139036
, 1.6963307783 
, 1.69752116996
, 1.69911077684
, 1.70105890286
, 1.70341035363
, 1.70576402908
, 1.70810242081
, 1.710642098
, 1.71314291212
, 1.71569484848
, 1.71834981821
, 1.7208678745 
, 1.69584139036
	],
	"MAE": [
	2.10286330728
, 1.42145905048
, 1.36302603517
, 1.34052092117
, 1.32853273024
, 1.32217378938
, 1.3188426712
, 1.31684463786
, 1.31631070655
, 1.31673337294
, 1.31707371849
, 1.31811492334
, 1.31939674466
, 1.32046729471
, 1.32212996192
, 1.3234545298
, 1.32568749157
, 1.32702883367
, 1.32921671117
, 1.33071198031
, 1.31631070655	
	]
};


conv_path = "conv.dat";
f = open(conv_path, 'r');
data = json.load(f);
f.close();

# evenly sampled time at 200ms intervals
epoch = range(1, 21);


## ML RMSE
f, ax_ml_rmse = plt.subplots()
ax_ml_rmse.set_xlim(1, 20)
ax_ml_rmse.xaxis.set_major_locator(LinearLocator(5));
ax_ml_rmse.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax_ml_rmse.set_ylim(0.8, 1.4)

ml_svd_rmse, = ax_ml_rmse.plot(epoch, data["MovieLens"]["SVD"]["RMSE"], 'r', linestyle = '-', label = "SVD")
ml_nmf_rmse, = ax_ml_rmse.plot(epoch, data["MovieLens"]["NMF"]["RMSE"], 'g', linestyle = '--', label = "NMF")
ml_fm_rmse, = ax_ml_rmse.plot(epoch, data["MovieLens"]["FM"]["RMSE"], 'y', linestyle = '-.',label = "FM")
ml_qsvd_rmse, = ax_ml_rmse.plot(epoch, ml_qsvd["RMSE"][:20], 'b', linestyle = ':', label = "SSVD")
ax_ml_rmse.legend(handles=[ml_svd_rmse, ml_nmf_rmse, ml_fm_rmse, ml_qsvd_rmse]);

ax_ml_rmse.set_xlabel(u"迭代周期数",fontproperties=zhfont1)
ax_ml_rmse.set_ylabel(u"RMSE指标",fontproperties=zhfont1)


## ML MAE
f, ax_ml_mae = plt.subplots()
ax_ml_mae.set_xlim(1, 20)
ax_ml_mae.xaxis.set_major_locator(LinearLocator(5));
ax_ml_mae.xaxis.set_major_formatter(FormatStrFormatter('%d'))


ml_svd_mae, = ax_ml_mae.plot(epoch, data["MovieLens"]["SVD"]["MAE"], 'r', linestyle = "-", label = "SVD")
ml_nmf_mae, = ax_ml_mae.plot(epoch, data["MovieLens"]["NMF"]["MAE"], 'g', linestyle = "--", label = "NMF")
ml_fm_mae, = ax_ml_mae.plot(epoch, data["MovieLens"]["FM"]["MAE"], 'y', linestyle = "-.", label = "FM")
ml_qsvd_mae, = ax_ml_mae.plot(epoch, ml_qsvd["MAE"][:20], 'b', linestyle = ':', label = "SSVD")
ax_ml_mae.legend(handles=[ml_svd_mae, ml_nmf_mae, ml_fm_mae, ml_qsvd_mae]);

ax_ml_mae.set_xlabel(u"迭代周期数",fontproperties=zhfont1)
ax_ml_mae.set_ylabel(u"MAE指标",fontproperties=zhfont1)


## BC RMSE
f, ax_bc_rmse = plt.subplots()
ax_bc_rmse.set_xlim(1, 20)
ax_bc_rmse.xaxis.set_major_locator(LinearLocator(5));
ax_bc_rmse.xaxis.set_major_formatter(FormatStrFormatter('%d'))

bc_svd_rmse, = ax_bc_rmse.plot(epoch, data["BookCrossing"]["SVD"]["RMSE"], 'r', linestyle = "-", label = "SVD")
bc_nmf_rmse, = ax_bc_rmse.plot(epoch, data["BookCrossing"]["NMF"]["RMSE"], 'g', linestyle = "--", label = "NMF")
bc_fm_rmse, = ax_bc_rmse.plot(epoch, data["BookCrossing"]["FM"]["RMSE"], 'y', linestyle = "-.", label = "FM")
bc_qsvd_rmse, = ax_bc_rmse.plot(epoch, bc_qsvd["RMSE"][:20], 'b', linestyle = ":", label = "SSVD")
ax_bc_rmse.legend(handles=[bc_svd_rmse, bc_nmf_rmse, bc_fm_rmse, bc_qsvd_rmse]);

ax_bc_rmse.set_xlabel(u"迭代周期数",fontproperties=zhfont1)
ax_bc_rmse.set_ylabel(u"RMSE指标",fontproperties=zhfont1)


## ML MAE
f, ax_bc_mae = plt.subplots()
ax_bc_mae.set_xlim(1, 20)
ax_bc_mae.xaxis.set_major_locator(LinearLocator(5));
ax_bc_mae.xaxis.set_major_formatter(FormatStrFormatter('%d'))


bc_svd_mae, = ax_bc_mae.plot(epoch, data["BookCrossing"]["SVD"]["MAE"], 'r', linestyle = "-", label = "SVD")
bc_nmf_mae, = ax_bc_mae.plot(epoch, data["BookCrossing"]["NMF"]["MAE"], 'g', linestyle = "--", label = "NMF")
bc_fm_mae, = ax_bc_mae.plot(epoch, data["BookCrossing"]["FM"]["MAE"], 'y', linestyle = "-.", label = "FM")
bc_qsvd_mae, = ax_bc_mae.plot(epoch, bc_qsvd["MAE"][:20], 'b', linestyle = ":", label = "SSVD")
ax_bc_mae.legend(handles=[bc_svd_mae, bc_nmf_mae, bc_fm_mae, bc_qsvd_mae]);

ax_bc_mae.set_xlabel(u"迭代周期数",fontproperties=zhfont1)
ax_bc_mae.set_ylabel(u"MAE指标",fontproperties=zhfont1)


plt.show();