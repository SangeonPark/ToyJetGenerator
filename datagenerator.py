import numpy as np
from pylorentz import Momentum4

class jet_data_generator(object):
    """    
    Input takes the following form. (massprior,quarkmass,nprong,nparticle)
    
    massprior : "signal" or "background" (signal to use Gaussian prior, backgroound for uniform)
    quarkmass : mass of the quark in the showering
    nprong : number of particles after hard splitting
    nparticle: total number of particles after showering 
    
    Then use generate_dataset(N) to generate N number of events  
    
    """
    def __init__(self, massprior, quarkmass, nprong, nparticle):
        super(jet_data_generator, self).__init__()
        self.massprior = massprior
        self.nprong = nprong
        self.quarkmass = quarkmass
        self.nparticle = nparticle
        self.zsoft = []
        self.zhard = []

    def reverse_insort(self, a, x, lo=0, hi=None):
        """Insert item x in list a, and keep it reverse-sorted assuming a
        is reverse-sorted. The key compared is the invariant mass of the 4-vector

        If x is already in a, insert it to the right of the rightmost x.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo+hi)//2
            if x.m > a[mid].m: hi = mid
            else: lo = mid+1
        a.insert(lo, x)


    def theta_to_eta(self,theta):
        if theta > np.pi:
            theta = 2*np.pi - theta

        return -np.log(np.tan(theta/2))


    def softsplit(self, mother):
        #Soft splitting performed in the rest frame of the mother, rotated, and then lorentz boosted back
        #Soft Splitting prior: Gaussian around 0  
        #print("mother = ",mother)
        randomdraw_theta = np.abs(np.random.normal(0,0.2))
        randomdraw_phi = np.random.uniform(0,2*np.pi)
        #print("randomdraw_theta = ",randomdraw_theta)
        #print("randomdraw_phi= ",   randomdraw_phi)
        #print("## mother m ## == ",mother.m)

        
        dau1_theta = mother.theta + randomdraw_theta
        dau2_theta = mother.theta - randomdraw_theta +np.pi


        dau1_phi = mother.phi + randomdraw_phi
        dau2_phi = mother.phi + randomdraw_phi + np.pi
        #print(dau1_phi, dau2_phi)
        dau1_m = np.random.uniform(0, mother.m/2)
        dau2_m = np.random.uniform(0, mother.m/2)

        dau1 = Momentum4.e_m_eta_phi(mother.m/2, dau1_m, self.theta_to_eta(dau1_theta), dau1_phi)
        dau2 = Momentum4.e_m_eta_phi(mother.m/2, dau2_m, self.theta_to_eta(dau2_theta), dau2_phi)
        #print("dau1, dau2", dau1, dau2)
        dau1 = dau1.boost_particle(mother)
        dau2 = dau2.boost_particle(mother)

        self.zsoft.append(np.min([dau1.p_t, dau2.p_t])/(dau1.p_t+dau2.p_t))
        return dau1, dau2

    def hardsplit(self, mother):
        #Hard splitting performed in the rest frame of the mother, rotated, and then lorentz boosted back
        #Hard splitting prior: Gaussian around pi/2,
        randomdraw_theta = np.abs(np.random.normal(np.pi/2,0.2))
        randomdraw_phi = np.random.uniform(0,2*np.pi)

        dau1_m = np.random.uniform(0, mother.m/2)
        dau2_m = np.random.uniform(0, mother.m/2)

        dau1_theta = mother.theta + randomdraw_theta
        dau2_theta = mother.theta - randomdraw_theta +np.pi

        dau1_phi = mother.phi + randomdraw_phi
        dau2_phi = mother.phi + randomdraw_phi + np.pi

        dau1 = Momentum4.e_m_eta_phi(mother.m/2, dau1_m, self.theta_to_eta(dau1_theta), dau1_phi)
        dau2 = Momentum4.e_m_eta_phi(mother.m/2, dau2_m, self.theta_to_eta(dau2_theta), dau2_phi)

        dau1 = dau1.boost_particle(mother)
        dau2 = dau2.boost_particle(mother)

        self.zhard.append(np.min([dau1.p_t, dau2.p_t])/(dau1.p_t+dau2.p_t))

        return dau1, dau2


    def draw_first_particle(self):
        #Draw mass from pdf
        if self.massprior == "signal":
            m = np.random.normal(80,4)


        if self.massprior == "background":
            m = np.random.uniform(0,100)
        
        p = np.random.exponential(400)
        return Momentum4.m_eta_phi_p(m, np.inf, 0, p)

    def hard_decays(self):
        hardparticle_list = [self.draw_first_particle()]
        prong = 1
        while prong < self.nprong:

            dau1, dau2 = self.hardsplit(hardparticle_list[0])
            hardparticle_list.pop(0)
            self.reverse_insort(hardparticle_list, dau1)
            self.reverse_insort(hardparticle_list, dau2)
            prong += 1

        return hardparticle_list

    def shower(self):
        showered_list = self.hard_decays()
        total_particle = len(showered_list)
        
        while total_particle < self.nparticle:
            
            dau1, dau2 = self.softsplit(showered_list[0])

            showered_list.pop(0)
            
            self.reverse_insort(showered_list, dau1)
            self.reverse_insort(showered_list, dau2)
   

            total_particle +=1

        return showered_list


    def generate_dataset(self, nevent):

        #output = torch.FloatTensor([])

        data = np.empty([nevent, 3*self.nparticle], dtype=float)

        for i in range(nevent):
            showered_list = self.shower()
            for j in range(self.nparticle):
                data[i,3*j] = showered_list[j].p_t 
                data[i,3*j+1] = showered_list[j].eta
                data[i,3*j+2] = showered_list[j].phi
        

        #return output
        return data


    #def visualize_one_event(self):


