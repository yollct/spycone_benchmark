import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys

sys.path.insert(0, "/Users/chit/Desktop/phd_scripts")
from spycone import dataset
from spycone import BioNetwork
from spycone import clustering
from spycone import preprocess
#from stat.pq_val import _cal_pqvals
#from enrichment.goterms import go_enrich
from spycone import iso_function
#import spycone as spy


import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)
from datetime import date

today = date.today()
md = today.strftime("%m%d")

#######params here
reps1 = 4
ng = 10000
tp=10
noises = [1,5,10]
###################

exec(open('/Users/chit/Desktop/phd_scripts/simulation/currentbest_nbinom.py').read())
exec(open('/Users/chit/Desktop/phd_scripts/simulation/currentbest_nbinom_equal.py').read())
funcd = {"maj":HMM, "eq":eqHMM}
import sys

p_trans = np.array([[0.9, 0.1, 0.2,0.8]])
p_init = np.array([0.9,0.1, 0.1,0.9])
for modeltype in ["maj","eq"]:
    results = pd.DataFrame(columns=['tools', 'group','tp', 'ngenes', 'noise','precision', 'recall', 'param'])
    func=funcd[modeltype]
    for noise in noises:
        print(modeltype)
        #hmm = func(ngenes=ng, nclusters=20, nobs=tp, p_trans=p_trans, p_init=p_init, reps=reps1, noise_level=noise, n_states=20)
        #hmm.simulation()
        #data = hmm.output
        #genest = hmm.isoformsdata
        data = pd.read_csv(f"zen_simdata_{modeltype}_{noise}.csv") 

        groundtruth = data[data['switched']=='yes'].isoforms.tolist()

        sim1 = dataset(ts=data.iloc[:,4:],
                gene_id = data['genes'],
                species=9606,
                keytype="entrezgeneid",
                transcript_id = data['isoforms'],
                symbs=data['genes'],
                timepts=tp,
                reps1=reps1)
        preprocess(sim1)
        data.to_csv(f"simdata_{modeltype}_{noise}.csv", index=False)
        print(data)

        print("done")
        print("running TSIS")
        import subprocess
        subprocess.call("Rscript tsis.R --args {} {} {} {}".format(tp-1, reps1, modeltype, str(noise)), shell=True)

        ##compare to tsis
        tsis_res = pd.read_csv("/Users/chit/Desktop/spycone/tsis_iso.csv", sep=" ")
        mtsis_res = pd.read_csv("/Users/chit/Desktop/spycone/major_tsis_iso.csv", sep=" ")
        spl_tsis_res = pd.read_csv("/Users/chit/Desktop/spycone/spline_tsis_iso.csv", sep=" ")
        spl_mtsis_res = pd.read_csv("/Users/chit/Desktop/spycone/spline_major_tsis_iso.csv", sep=" ")

        print("running ticone as")
        iso = iso_function(sim1)
        res = iso.detect_isoform_switch(corr_cutoff=0, event_im_cutoff = 0, min_diff = 0, p_val_cutoff = 1, spline=False)
        res.to_csv("/Users/chit/Desktop/spycone/spy_iso.csv")
        spl_res = iso.detect_isoform_switch(corr_cutoff=0, event_im_cutoff = 0, min_diff = 0, p_val_cutoff = 1, spline=True)
        spl_res.to_csv("/Users/chit/Desktop/spycone/spline_spy_iso.csv")

        tsdiff = np.arange(0, round(np.max(mtsis_res['diff'].to_numpy()),2), round(np.max(mtsis_res['diff'].to_numpy()),2)/50)
        tscorr = np.arange(-1,1,2/50)
        mtscorr = np.arange(-1,1,2/50)
        spycorr = np.arange(0,1,1/50)
        spydiff = np.arange(0,round(np.max(res['diff'].to_numpy()),2), round(np.max(res['diff'].to_numpy()),2)/50)
        splspycorr = np.arange(0,1,1/50)
        splspydiff = np.arange(0,round(np.max(spl_res['diff'].to_numpy()),2), round(np.max(spl_res['diff'].to_numpy()),2)/50)
        print(f"max diff: {round(np.max(res['diff'].to_numpy()),2)}")
        print(spydiff)
        for t in range(0,len(tsdiff)-1):
            tsis_ress = tsis_res.loc[(tsis_res['diff'] > tsdiff[t])]
            mtsis_ress = mtsis_res.loc[(mtsis_res['diff'] > tsdiff[t])]
            spl_tsis_ress = spl_tsis_res.loc[(spl_tsis_res['diff'] > tsdiff[t])]
            spl_mtsis_ress = spl_mtsis_res.loc[(spl_mtsis_res['diff'] > tsdiff[t])]

            ress = res.loc[(res['adj_pval'] < 0.05)  &  (res['corr'] > 0.5) & (res['diff'] > spydiff[t]) & (res['event_importance']>0)]
            spl_ress = spl_res.loc[(spl_res['adj_pval'] < 0.05)  &  (spl_res['corr'] > 0.5) & (spl_res['diff'] > splspydiff[t]) & (spl_res['event_importance']>0)]
            simiso = np.unique(ress['major_transcript'].unique().tolist() + ress['minor_transcript'].unique().tolist())
            spl_simiso = np.unique(spl_ress['major_transcript'].unique().tolist() + spl_ress['minor_transcript'].unique().tolist())
            tsiso = np.unique(tsis_ress['iso1'].unique().tolist()+tsis_ress['iso2'].unique().tolist())
            mtsiso = np.unique(mtsis_ress['iso1'].unique().tolist()+mtsis_ress['iso2'].unique().tolist())
            spl_tsiso = np.unique(spl_tsis_ress['iso1'].unique().tolist()+spl_tsis_ress['iso2'].unique().tolist())
            spl_mtsiso = np.unique(spl_mtsis_ress['iso1'].unique().tolist()+spl_mtsis_ress['iso2'].unique().tolist())

            #confusion matrix
            y_true = [1 if i =="yes" else 0 for i in data['switched'] ]
            y_pred = [1 if i in simiso else 0 for i in data['isoforms']]

            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import precision_recall_curve
            tn, fp, fn, tpp = confusion_matrix(y_true, y_pred).ravel()
            ticprecision, ticrecall = tpp / (tpp+fp), tpp/(tpp+fn)
            ticprecision, ticrecall

            splspy_pred = [1 if i in spl_simiso else 0 for i in data['isoforms']]

            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import precision_recall_curve
            tn, fp, fn, tpp = confusion_matrix(y_true, splspy_pred).ravel()
            spl_ticprecision, spl_ticrecall = tpp / (tpp+fp), tpp/(tpp+fn)

            tsis_pred = [1 if i in tsiso else 0 for i in data['isoforms']]
            tn, fp, fn, tpp = confusion_matrix(y_true, tsis_pred).ravel()
            tsisprecision, tsisrecall = tpp / (tpp+fp), tpp/(tpp+fn)
            if np.isnan(tsisprecision):
                tsisprecision = 0

            mtsis_pred = [1 if i in mtsiso else 0 for i in data['isoforms']]
            tn, fp, fn, tpp = confusion_matrix(y_true, mtsis_pred).ravel()
            mtsisprecision, mtsisrecall = tpp / (tpp+fp), tpp/(tpp+fn)
            if np.isnan(mtsisprecision):
                mtsisprecision = 0

            spl_tsis_pred = [1 if i in spl_tsiso else 0 for i in data['isoforms']]
            tn, fp, fn, tpp = confusion_matrix(y_true, spl_tsis_pred).ravel()
            spl_tsisprecision, spl_tsisrecall = tpp / (tpp+fp), tpp/(tpp+fn)
            if np.isnan(spl_tsisprecision):
                spl_tsisprecision = 0

            spl_mtsis_pred = [1 if i in spl_mtsiso else 0 for i in data['isoforms']]
            tn, fp, fn, tpp = confusion_matrix(y_true, spl_mtsis_pred).ravel()
            spl_mtsisprecision, spl_mtsisrecall = tpp / (tpp+fp), tpp/(tpp+fn)
            if np.isnan(spl_mtsisprecision):
                spl_mtsisprecision = 0
            

            results.loc[results.shape[0]] = ['Spycone',f'spycone_{noise}', tp, ng, noise, ticprecision, ticrecall, spydiff[t]]
            results.loc[results.shape[0]] = ['Spline_Spycone',f'spline_spycone_{noise}', tp, ng, noise, spl_ticprecision, spl_ticrecall, splspydiff[t]]
            results.loc[results.shape[0]] = ['TSIS', f'tsis_{noise}', tp, ng, noise, tsisprecision, tsisrecall, tsdiff[t]]
            results.loc[results.shape[0]] = ['major_TSIS',f'major_tsis_{noise}', tp, ng, noise, mtsisprecision, mtsisrecall, tsdiff[t]]
            results.loc[results.shape[0]] = ['spline_TSIS', f'spline_tsis_{noise}', tp, ng, noise, spl_tsisprecision, spl_tsisrecall, tsdiff[t]]
            results.loc[results.shape[0]] = ['major_spline_TSIS',f'spline_major_tsis_{noise}', tp, ng, noise, spl_mtsisprecision, spl_mtsisrecall, tsdiff[t]]


    results.to_csv("/Users/chit/Desktop/spycone/for_prcurve.csv", index=False)

    cuv = results
    maxcuv = cuv.iloc[cuv.groupby('group')['recall'].idxmax(),:]

    markers = ['o','*','X','p','D','P','>','<','H']
    plt.style.use('seaborn-colorblind')
    cols = ["#F49335", "#D085F4", "#2574F4", "#F42766", "#28F584", "#a474dc"]
    print(len(cols))
    fig, ax = plt.subplots(1, len(noises), figsize=(15,4))

    import matplotlib.lines as mlines

    hdls = []

    for n, subax in enumerate(ax):
        for i,t in enumerate(cuv['tools'].unique()):
            tmp = maxcuv[(maxcuv['noise']==noises[n]) & (maxcuv['tools']==t)]
            sub = cuv[(cuv['noise']==noises[n]) & (cuv['tools']==t)]

            pp, = subax.plot(sub['recall'], sub['precision'], c=cols[i])
            subax.scatter(tmp.recall, tmp.precision, marker=markers[i], s=120, c=cols[i])
            hdls.append(pp)
            subax.set_xlim((0,1))
            subax.set_ylim((0,1.05))
            subax.set_yticks([])
            if modeltype =="maj":
                subax.set_xticks([])
                subax.set_title(f"Noise level {noises[n]}", fontsize=12)
            
    for i,t in enumerate(cuv['tools'].unique()):
        hdls.append(mlines.Line2D([], [], marker=markers[i],
                                markersize=13, label=t, color=cols[i]))

    ax[0].set_ylabel("Model 1 \n Precision", fontsize=15)
    ax[0].set_yticks(np.arange(0,1.1,0.2))

    if modeltype == "eq":
        ax[0].set_ylabel("Model 2 \n Precision", fontsize=15)
        ax[len(noises)-1].legend(handles=hdls, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        ax[1].set_xlabel("Recall")

    plt.savefig(f"/Users/chit/Desktop/talk/spline_prc_nbinom_{md}_{modeltype}.svg", bbox_inches="tight", device="svg")
    plt.close()

##model 1

# import sys

# results = pd.DataFrame(columns=['tools', 'group','tp', 'ngenes', 'noise','precision', 'recall', 'param'])

# p_trans = np.array([[0.9,0.1],[0.2,0.8]])
# p_init = np.array([0.5,0.5])
# for noise in noises:
#     hmm = HMM(ngenes=ng, nclusters=20, nobs=tp, p_trans=p_trans, p_init=p_init, reps=reps1, noise_level=noise)
#     hmm.simulation()
#     data = hmm.output
#     genest = hmm.isoformsdata

#     groundtruth = data[data['switched']=='yes'].isoforms.tolist()

#     sim1 = spy.dataset(ts=data.iloc[:,4:],
#             gene_id = data['genes'],
#             species=9606,
#             keytype="entrezgeneid",
#             transcript_id = data['isoforms'],
#             symbs=data['genes'],
#             timepts=tp,
#             reps1=reps1)
#     spy.preprocess(sim1)
#     data.to_csv("/Users/chit/Desktop/spycone/simdata.csv", index=False)
#     print("done")
#     print("running TSIS")
#     import subprocess
#     subprocess.call ("Rscript /Users/chit/Desktop/spycone/tsis.R --args {} {}".format(tp-1, reps1), shell=True)

#     ##compare to tsis
#     tsis_res = pd.read_csv("/Users/chit/Desktop/spycone/tsis_iso.csv", sep=" ")
#     mtsis_res = pd.read_csv("/Users/chit/Desktop/spycone/major_tsis_iso.csv", sep=" ")

#     print("running ticone as")
#     iso = spy.iso_function(sim1)
#     res = iso.detect_isoform_switch(corr_cutoff=0, event_im_cutoff = 0, min_diff = 0, p_val_cutoff = 1)   

#     tsdiff = np.arange(0, 200, 200/50)
#     tscorr = np.arange(-1,1,2/50)
#     mtsdiff = np.arange(0, 200, 200/50)
#     mtscorr = np.arange(-1,1,2/50)
#     spycorr = np.arange(0,1,1/50)
#     spydiff = np.arange(0,round(np.max(res['diff'].to_numpy()),2), round(np.max(res['diff'].to_numpy()),2)/50)

#     for t in range(0,len(tsdiff)-1):
#         tsis_ress = tsis_res.loc[(tsis_res['diff'] > tsdiff[t])]
#         mtsis_ress = mtsis_res.loc[(mtsis_res['diff'] > mtsdiff[t])]

#         ress = res.loc[(res['adj_pval'] < 0.05)  &  (res['corr'] > 0.5) & (res['diff'] > spydiff[t]) & (res['event_importance']>0)]
#         simiso = np.unique(ress['major_transcript'].unique().tolist() + ress['minor_transcript'].unique().tolist())
#         tsiso = np.unique(tsis_ress['iso1'].unique().tolist()+tsis_ress['iso2'].unique().tolist())
#         mtsiso = np.unique(mtsis_ress['iso1'].unique().tolist()+mtsis_ress['iso2'].unique().tolist())

#         #confusion matrix
#         y_true = [1 if i =="yes" else 0 for i in data['switched'] ]
#         y_pred = [1 if i in simiso else 0 for i in data['isoforms']]

#         from sklearn.metrics import confusion_matrix
#         from sklearn.metrics import precision_recall_curve
#         tn, fp, fn, tpp = confusion_matrix(y_true, y_pred).ravel()
#         ticprecision, ticrecall = tpp / (tpp+fp), tpp/(tpp+fn)
#         ticprecision, ticrecall

#         tsis_pred = [1 if i in tsiso else 0 for i in data['isoforms']]
#         tn, fp, fn, tpp = confusion_matrix(y_true, tsis_pred).ravel()
#         tsisprecision, tsisrecall = tpp / (tpp+fp), tpp/(tpp+fn)
#         if np.isnan(tsisprecision):
#             tsisprecision = 0

#         mtsis_pred = [1 if i in mtsiso else 0 for i in data['isoforms']]
#         tn, fp, fn, tpp = confusion_matrix(y_true, mtsis_pred).ravel()
#         mtsisprecision, mtsisrecall = tpp / (tpp+fp), tpp/(tpp+fn)
#         if np.isnan(mtsisprecision):
#             mtsisprecision = 0


#         results.loc[results.shape[0]] = ['Spycone',f'spycone_{noise}', tp, ng, noise, ticprecision, ticrecall, spydiff[t]]
#         results.loc[results.shape[0]] = ['TSIS', f'tsis_{noise}', tp, ng, noise, tsisprecision, tsisrecall, tsdiff[t]]
#         results.loc[results.shape[0]] = ['major_TSIS',f'major_tsis_{noise}', tp, ng, noise, mtsisprecision, mtsisrecall, mtsdiff[t]]

# results.to_csv("/Users/chit/Desktop/spycone/for_prcurve1.csv", index=False)

# cuv = results
# maxcuv = cuv.iloc[cuv.groupby('group')['recall'].idxmax(),:]

# markers = ['o','*','X','p','D','P','>','<','H']
# plt.style.use('seaborn-colorblind')
# cols = ["#505BEA","#FB9C2D","#76DACA"]

# print(len(cols))
# fig, ax = plt.subplots(1, len(noises), figsize=(15,4))

# import matplotlib.lines as mlines

# hdls = []

# for n, subax in enumerate(ax):
#     for i,t in enumerate(cuv['tools'].unique()):
#         tmp = maxcuv[(maxcuv['noise']==noises[n]) & (maxcuv['tools']==t)]
#         sub = cuv[(cuv['noise']==noises[n]) & (cuv['tools']==t)]

#         pp, = subax.plot(sub['recall'], sub['precision'], c=cols[i])
#         subax.scatter(tmp.recall, tmp.precision, marker=markers[i], s=120, c=cols[i])
#         hdls.append(pp)
#         subax.set_xlim((0,1))
#         subax.set_ylim((0,1.05))
#         subax.set_yticks([])
#         #subax.set_title(f"Model 2 - noise level {round(1/noises[n],2)}", fontsize=12)
#         #subax.set_xlabel("Recall", fontsize=15)


# for i,t in enumerate(cuv['tools'].unique()):
#     hdls.append(mlines.Line2D([], [], marker=markers[i],
#                             markersize=13, label=t, color=cols[i]))

# ax[2].legend(handles=hdls, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# ax[0].set_ylabel("Model 2 \n Precision", fontsize=15)
# ax[0].set_yticks(np.arange(0,1.1,0.2))
# ax[1].set_xlabel("Recall", fontsize=15)

# plt.savefig(f"/Users/chit/Desktop/talk/prc_nbinom_{md}_eq.svg", bbox_inches="tight", device="svg")