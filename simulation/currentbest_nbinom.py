import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import dirichlet, multinomial, norm, invgamma, poisson, beta, multivariate_normal, halfcauchy, halfnorm, gamma
import random
import seaborn as sns
from scipy.stats.mstats import mquantiles
import json

ref_var = json.load(open("/Users/chit/Desktop/phd_scripts/ref_var.pkl","r"))

def unconditionalProbability(Ptrans):
    '''Compute the unconditional probability for the states of a
       Markov chain.'''

    m = Ptrans.shape[0]
    P = np.column_stack((Ptrans, 1. - Ptrans.sum(axis=1)))

    I = np.eye(m)
    U = np.ones((m, m))
    u = np.ones(m)
    return(np.linalg.solve((I - P + U).T, u))

class HMM():
    def __init__(self, ngenes, nclusters, nobs, reps=3, n_states=5, p_init=None, p_trans=None, isoformsdata=None, output=None, noise_level=0.5, estimated_var=None):
        self.p_init = p_init
        self.p_trans = p_trans
        self.ngenes = ngenes
        self.nclusters =nclusters
        self.nobs = nobs
        self.isoformsdata = isoformsdata
        self.output = output
        self.reps = reps
        self.n_states = n_states
        self.noise_level = noise_level
        self.abun = []
        self.esitmated_var = estimated_var
        
    def isoform_def(self):
        genes =[]
        isoforms = []
        clusters = []
        nisoforms = []
        for i in range(self.ngenes):
            n_isoform = poisson.rvs(4,size=1)
            if n_isoform == 0:
                n_isoform = np.array([1])
            
            genes.append(f'g{i}')
            clusters.append(random.randint(0,self.nclusters-1))
            nisoforms.append(n_isoform[0])
            
        tmp = pd.DataFrame({'genes':genes, 'clusters':clusters, 'niso':nisoforms})
        self.isoformsdata = tmp.sort_values('clusters')
    
    def gene_states(self):
        self.p_trans = dirichlet.rvs(np.ones(4), size=4)
        self.p_init = unconditionalProbability(self.p_trans[:,:-1])
        
        init_state = np.where(multinomial.rvs(1, self.p_init)==1)[0]
        gene_state = np.array(init_state)

        for j in range(self.nobs):
            p_tr = self.p_trans[gene_state[-1]]
            new_state = np.where(multinomial.rvs(1, p_tr)==1)[0]
            gene_state = np.append(gene_state, new_state)
            
        return gene_state[1:]
    
    def gene_exp_states(self):
        '''
        if gene expression is needed
        '''
        p_trans = dirichlet.rvs(np.ones(self.n_states), size=self.n_states)
        p_init = unconditionalProbability(self.p_trans[:,:-1])

        init_state = np.where(multinomial.rvs(1, self.p_init)==1)[0]
        gene_state = np.array(init_state)

        for j in range(self.nobs):
            p_tr = p_trans[gene_state[-1]]
            new_state = np.where(multinomial.rvs(1, p_tr)==1)[0]
            gene_state = np.append(gene_state, new_state)
            
        return gene_state[1:]
    
    
    def emission(self, thisstate):
        #r for each states
        r0= np.arange(1, self.n_states+1) * 10
        
        iso_dist = np.array([], dtype="float32")
        for x,y in enumerate(thisstate):
            r = r0[y]
            this_y = nbinom.rvs(r, 0.25)
            iso_dist = np.append(iso_dist, this_y)
        return iso_dist
    
    
    
#     def add_noise(self, noise_ratio, noise_level):
#         n_noise = int(self.output.shape[0])*noise_ratio
        
#         for _ in range(int(n_noise)):
#             #low or high expressed genes without isoform switching
# #             exp = np.where(multinomial.rvs(1, np.array([0.5,0.5]))==1)[0]
# #             if exp == 0:
#             noise_emi = norm.rvs(2, noise_level, size=self.nobs*self.reps)
#             noise = 'low noise'
            
#             noise_emi[noise_emi<0] = 0
#             gene = np.random.choice(self.output['genes'].unique())
            
#             niso = self.output[self.output['genes']==gene].shape[0]
            
#             self.output.loc[self.output.shape[0]] = [noise, 
#                                                      gene, 
#                                                      '{}i{}'.format(gene,niso+1), 'no'] + list(noise_emi)

    def swap_random(self, arr):
        idx = range(len(arr))
        if len(idx)!=2:     
            arr1 = arr.copy()
            i1, i2 = random.sample(idx, 2)
        else:
            try:
                arr1 = arr.copy()
                i1=0
                i2=1
            except:
                print(f"exception: {arr}")
        return 0, 1

    def get_var(self, x):
        if x < 0.1:
            x=x+0.2
        tmp=0
        while tmp==0:
            tmp = gamma.rvs(x)
        return tmp
      
        
    def simulation(self):
        #define gene and correspond isoforms and clusters
        self.isoform_def()
        iso_states = []
        gene_emi=[]
        col = ['clusters', 'genes', 'isoforms','switched'] +  ['t{}_{}'.format(x,y) for y in range(self.reps) for x in range(self.nobs)]
        output = pd.DataFrame(columns = col)
        
        #model for each gene in each cluster
        
        for i in range(self.nclusters):
            df = self.isoformsdata[self.isoformsdata['clusters']==i]
            
            #this states depict the isoforms distribution changes
            cluster_state = self.gene_states()
            #this states depict the expression level changes
            cluster_exp_state = self.gene_exp_states()
            #with emissions
            

            for _ in range(df.shape[0]):
                iso_states.append(cluster_state)
                gene_emi.append(cluster_exp_state)
        self.isoformsdata['states'] = iso_states
        self.isoformsdata['emission'] = gene_emi

        for j in range(self.ngenes):
            gene = self.isoformsdata[self.isoformsdata['genes']=='g{}'.format(j)].reindex()
            # if (0 not in gene['states'].tolist()) & (1 not in gene['states'].tolist()):
            #     idx = random.sample(range(1, self.reps-1),1)
            #     gene['states'].tolist()[0][idx]=1

            cluster_exp_emi = self.emission(gene['emission'].tolist()[0])

            niso = int(gene['niso'])
            iso_emi = np.zeros((niso, self.nobs*self.reps))
            
            #overall expression gene mean
            if niso>1:
                gene_mean = random.sample(ref_var[str(niso)].keys(), 1)[0]

            if niso >= 3:
                n_switched = np.random.choice([2])
                #set the iso dist for each state 
                iso_switch_prob = np.arange(1, (int(n_switched))+1)*20
#                 iso_dist0 = dirichlet.rvs(alpha=np.append([1]*(niso-n_switched), iso_switch_prob))[0]
#                 iso_dist1 = dirichlet.rvs(np.append([1]*(niso-n_switched), self.swap_random(iso_switch_prob)))[0]
                iso_dist0 = np.append(iso_switch_prob, [1]*(niso-n_switched))
                i1, i2= self.swap_random(iso_switch_prob)

                ##model transcript abundance -> transcript expression
                iso_abun0 = dirichlet.rvs(iso_dist0)[0] 
                
                iso_abun1 = iso_abun0.copy()
                iso_abun0[i1], iso_abun0[i2] = iso_abun1[i2], iso_abun1[i1] 
                iso_emi0 = iso_abun0 * float(gene_mean) 
                iso_emi1 = iso_abun1 * float(gene_mean)
                vars0 = [self.get_var((iso_emi0[x]*self.noise_level)/(self.noise_level+iso_emi0[x])) for x in range(len(iso_abun0))]
                vars1 = [self.get_var((iso_emi1[x]*self.noise_level)/(self.noise_level+iso_emi1[x])) for x in range(len(iso_abun0))]

                checkswitched=False
                for r in range(self.reps):
                    for x,n in enumerate(gene['states'].tolist()[0]):
                        if (n == 0) or (n==1):
                            tmp = [norm.rvs(iso_emi0[x], vars0[x])  for x in range(len(iso_abun0))]
                            iso_emi[:,x+(self.nobs*r)] = tmp
                            #print(gene_mean, niso, iso_abun0, vars)
                            
                        else:
                            checkswitched=True
                            tmp = [norm.rvs(iso_emi1[x], vars1[x]) for x in range(len(iso_abun1))]
                            iso_emi[:,x+(self.nobs*r)] = tmp 
                            #print(gene_mean, niso, iso_abun1, vars)
                switched = ['no']*niso
                if checkswitched:
                    switched[i1], switched[i2] = "yes", "yes"

            elif niso > 1:
                
                #set the iso dist for each state    
                iso_dist0 = [1]*niso
                i1, i2 = self.swap_random(iso_dist0)
                
                #abun
                iso_abun0 = dirichlet.rvs(iso_dist0)[0]
                iso_abun1 = iso_abun0.copy()
                iso_abun0[i1], iso_abun0[i2] = iso_abun1[i2], iso_abun1[i1] 
                iso_emi0 = iso_abun0 * float(gene_mean)
                iso_emi1 = iso_abun1 * float(gene_mean)
                vars0 = [self.get_var((iso_emi0[x]*self.noise_level)/(self.noise_level+iso_emi0[x])) for x in range(len(iso_abun0))]
                vars1 = [self.get_var((iso_emi1[x]*self.noise_level)/(self.noise_level+iso_emi1[x])) for x in range(len(iso_abun0))]
                checkswitched=False
                for r in range(self.reps):
                    for x,n in enumerate(gene['states'].tolist()[0]):
                        if (n == 0) or (n==1):
                            tmp = [norm.rvs(iso_emi0[x], vars0[x])  for x in range(len(iso_abun0))]
                            iso_emi[:,x+(self.nobs*r)] = tmp
                        else:
                            checkswitched=True
                            tmp = [norm.rvs(iso_emi1[x], vars1[x]) for x in range(len(iso_abun1))]
                            iso_emi[:,x+(self.nobs*r)] = tmp 


                switched = ['no']*niso
                if checkswitched:
                    switched = ['yes']*niso

            else:
                iso_emi=np.expand_dims(np.array(norm.rvs(1,1e-1, size=self.nobs*self.reps)),axis=0)
                self.abun.append("none")
                switched = ['no']*niso

            iso_emi = iso_emi+np.abs(np.min(iso_emi))
            for ni in range(iso_emi.shape[0]):
                isoemi = iso_emi[ni]
                
                #isoemi = np.multiply(iso_emi[ni], np.tile(cluster_exp_emi, self.reps))
                #isoemi = np.multiply(iso_emi[ni], np.tile(norm.rvs(20, 0.1, size= self.nobs), self.reps))
   
                output_row = [gene['clusters'].tolist()[0], 
                            f'g{j}', 
                            f'g{j}i{ni}',
                            switched[ni]] + isoemi.tolist()

                                                               
                
                output.loc[output.shape[0]] = output_row
            
                
        output.iloc[:,4:] = output.iloc[:,4:] + np.abs(output.iloc[:,4:].min().min())

        self.output = output
        #self.add_noise(1, self.noise_level)
        return output
    


