import numpy as np
import multiprocessing as mp
from pylorentz import Momentum4
from scipy.stats import beta
import multiprocessing as mp
from multiprocessing import Process, Pool
from dataclasses import dataclass
import math

@dataclass
class particle:
    mom: Momentum4
    randtheta: float
    z: float
    m1: float
    m2: float

class jet_data_generator(object):
    """    
    Input takes the following form. (massprior,quarkmass,nprong,nparticle)
    
    massprior : "signal" or "background" (signal to use Gaussian prior, backgroound for uniform)
    quarkmass : mass of the quark in the showering
    nprong : number of particles after hard splitting
    nparticle: total number of particles after showering 
    
    Then use generate_dataset(N) to generate N number of events  
    
    """
    def __init__(self, massprior, nprong, nparticle, doFixP,       doMultiprocess=False, ncore = 0):
        super(jet_data_generator, self).__init__()
        self.massprior = massprior
        self.nprong = nprong
        #self.quarkmass = quarkmass
        self.nparticle = nparticle
        self.zsoft = []
        self.zhard = []
        self.z = []
        self.randtheta = []
        self.doFixP = doFixP
        self.doMultiprocess = doMultiprocess
        self.ncore = ncore

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
            if (x.mom.m > a[mid].mom.m and x.mom.p > 1) or  (x.mom.p >  a[mid].mom.p and x.mom.p < 1): hi = mid
            else: lo = mid+1
        a.insert(lo, x)

    def rotation_matrix(self,axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        
    def theta_to_eta(self,theta):
        if theta > np.pi:
            theta = 2*np.pi - theta
        return -np.log(np.tan(theta/2))

    def sintheta1(self,z,theta):
        return (-z/(1-z))*np.sin(theta)

    def mass(self,z,theta):
        sint1=self.sintheta1(z,theta)
        cost1=np.sqrt(1-sint1**2)
        p=z*np.cos(theta)+(1-z)*cost1
        return np.sqrt(1-p**2)

    def gamma(self,z,theta):
        return 1./self.mass(z,theta)

    def betaval(self,gamma):
        return np.sqrt(1-1/gamma**2)

    def sinthetaR(self,z,theta):
        gammavar=self.gamma(z,theta)
        K=gammavar*np.tan(theta)
        betavar=self.betaval(gammavar)
        return 2*K/(K**2+betavar)

    def sinthetaR2(self,z,theta):#more robust solution
        mom=(self.mass(z,theta)/2)
        return z*np.sin(theta)/mom
    
    def restmom(self,z,theta):
        sintR=self.sinthetaR(z,theta)
        gammavar=self.gamma(z,theta)
        betavar=self.betaval(gammavar)
        return z*betavar*np.sin(theta)/sintR

    def randmtheta(self,zmin=1e-5,thetamin=1e-5,thetamax=0.4,mmin=1e-5,mmax=1e3,isize=1):
        z = beta.rvs(0.1,1, size=isize)
        theta = beta.rvs(0.1,1, size=isize)
        sinthetarest=self.sinthetaR2(z,theta)
        restmass=self.mass(z,theta)
        count=0
        while(theta < thetamin or theta > thetamax or
              restmass < mmin  or restmass > mmax  or 
              np.isnan(sinthetarest) or np.abs(sinthetarest) > 1 or
              z < zmin):
            z = beta.rvs(0.1,1, size=isize)
            theta = beta.rvs(0.1,1, size=isize)
            sinthetarest=self.sinthetaR2(z,theta)
            restmass=self.mass(z,theta)
            count+=1
            if count > 1000:
                #sample [0.00761592] - 1e-05 < [0.00122907] 0.0009174540921386668 < [1.48784275e-05] 1e-05 < [0.31999996] < 0.4
                print("sample",sinthetarest,"-",mmin,"<",restmass,"<",mmax,zmin,"<",z,thetamin,"<",theta,"<",thetamax)
        #print("z:",z,"theta:",theta,"thetarest",sinthetarest)
        return restmass,sinthetarest

    def p2(self,iM1,iM2,iMM):
        """       
        #Phil, fix to ensure momentum is back to back and mass is perserved
        #sqrt(p^2+m1^2)+sqrt(p^2+m2^2)=mother.mom.m=> solve for p^2
        """
        return (iM1**4+iM2**4+iMM**4-2*(iM1*iM2)**2-2*(iM1*iMM)**2-2*(iM2*iMM)**2)/(2*iMM)**2

    def rotateTheta(self,idau,itheta):
        v1     = [idau.p_x,idau.p_y,idau.p_z]
        axis=[0,1,0]
        v1rot=np.dot(self.rotation_matrix(axis,itheta), v1)
        dau1_mom = Momentum4(idau.e,v1rot[0],v1rot[1],v1rot[2])
        return dau1_mom

    def rotatePhi(self,idau,itheta):
        v1     = [idau.p_x,idau.p_y,idau.p_z]
        axis=[1,0,0]
        v1rot=np.dot(self.rotation_matrix(axis,itheta), v1)
        dau1_mom = Momentum4(idau.e,v1rot[0],v1rot[1],v1rot[2])
        return dau1_mom
 
    def softsplit_old(self, mother):
        #Soft splitting performed in the rest frame of the mother, rotated, and then lorentz boosted back
        #Soft Splitting prior: Gaussian around 0  
        np.random.seed()
        randomdraw_phi = np.random.uniform(0,2*np.pi)
        randomdraw_theta=mother.randtheta
        if randomdraw_theta == -1000:
            _,randomdraw_theta=self.randmtheta(zmin=0.2/mother.mom.p,isize=1)
        #sample next event
        #dau1_m,randtheta1 = self.randmtheta(zmin=0.2/mother.mom.p,mmin=mother.mom.m/2/mother.mom.p,isize=1)
        #dau2_m,randtheta2 = self.randmtheta(zmin=0.2/mother.mom.p,mmin=mother.mom.m/2/mother.mom.p,isize=1)
        dau1_m,randtheta1 = self.randmtheta(zmin=0.2/mother.mom.p,mmin=1e-10,mmax=mother.mom.m/2/mother.mom.p,isize=1)
        dau2_m,randtheta2 = self.randmtheta(zmin=0.2/mother.mom.p,mmin=1e-10,mmax=mother.mom.m/2/mother.mom.p,isize=1)
        dau1_m = dau1_m[0]*mother.mom.p #mass was dimensionless in sampler
        dau2_m = dau2_m[0]*mother.mom.p
        dau_p2   = self.p2(dau1_m,dau2_m,mother.mom.m)
        dau1_e   = np.sqrt(dau_p2+dau1_m**2)
        dau2_e   = np.sqrt(dau_p2+dau2_m**2)
        corr=(mother.mom.m/2)/np.sqrt(dau_p2)#correction to ensure transverse momentum is z*p*sin(theta), factor is 1 for massless particles
        corr=1
        dau1_theta = (np.pi/2 + np.arcsin(randomdraw_theta*corr)) 
        dau2_theta = (np.pi/2 - np.arcsin(randomdraw_theta*corr))
        dau1_phi = mother.mom.phi + randomdraw_phi
        dau2_phi = mother.mom.phi + randomdraw_phi + np.pi
        dau1_phi %= (2*np.pi)
        dau2_phi %= (2*np.pi)
        #prep for 4-vector
        eta1   = self.theta_to_eta(dau1_theta)
        eta2   = self.theta_to_eta(dau2_theta)
        #print("eta1:",eta1,"eta2:",eta2,"dsintheta",randomdraw_theta,"corr",corr,"-",dau1_m,dau2_m,"-",np.sqrt(dau_p2),mother.mom.m/2)
        dau1_mom = Momentum4.e_m_eta_phi(dau1_e, dau1_m, eta1[0], dau1_phi)
        dau2_mom = Momentum4.e_m_eta_phi(dau2_e, dau2_m, eta2[0], dau2_phi)        
        dau1_mom = self.rotateTheta(dau1_mom,mother.mom.theta)
        dau2_mom = self.rotateTheta(dau2_mom,mother.mom.theta)
        #finally boost and send it off
        dau1 = particle(mom=dau1_mom,randtheta=randtheta1)
        dau2 = particle(mom=dau2_mom,randtheta=randtheta2)
        dau1.mom = dau1.mom.boost_particle(mother.mom)
        dau2.mom = dau2.mom.boost_particle(mother.mom)
        #print("deta:",np.abs(mother.mom.eta-dau1.mom.eta),mother.randtheta)
        self.z.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.zsoft.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.randtheta.append(randomdraw_theta)
        return dau1, dau2

    def massapprox(self,z,theta,p):
        return np.sqrt(z*(1-z))*p*theta

    def massp(self,z,theta,p):
        return self.mass(z,theta)*p

    def dau2mass(self,mother,z,theta):
        dau1_m = self.massapprox(z,theta,mother.mom.p)
        dau1_e = (mother.mom.p)*mother.z
        dau1_eta = self.theta_to_eta(-mother.randtheta+np.pi/2)
        print(dau1_eta,dau1_m,dau1_e)
        d1=Momentum4.e_m_eta_phi(dau1_e[0], dau1_m, dau1_eta[0],0)
        mo=Momentum4.e_m_eta_phi(mother.mom.e,mother.mom.m, 0,0)
        print(dau1_e,dau1_m,mo,d1,(mo-d1).m,mo.m,d1.m)
        return ((mo-d1).m)/mother.mom.p

    def randztheta(self,mother,zmin=1e-5,thetamin=1e-5,thetamax=0.4,mmin=1e-10,mmax=0.2,isize=1):
        z = beta.rvs(0.1,1, size=isize)
        theta = beta.rvs(0.1,1, size=isize)
        restmass=self.mass(z,theta)
        dau2mass=self.dau2mass(mother,z,theta)
        count=0
        massmax = np.minimum(0.2,mmax)
        while(theta < thetamin or theta > thetamax    or
              restmass < mmin  or restmass > massmax  or
              dau2mass < mmin  or dau2mass > massmax  or
              z < zmin or z > 0.5):
            z = beta.rvs(0.1,1, size=isize)
            theta = beta.rvs(0.1,1, size=isize)
            restmass=self.mass(z,theta)
            dau2mass=self.dau2mass(mother,z,theta)
            count+=1
            if count > 1000:
                print("theta 1 ",restmass,massmax,z,theta,dau2mass,mother.z,mother.randtheta)
        print("dau2mass",restmass*mother.mom.p,dau2mass*mother.mom.p,mother.randtheta)
        #restmass=self.mass(z,theta)
        #massapprox=self.massapprox(z,theta,1)
        #print("mass check",restmass,massapprox,"---",z,theta,zmin,mmin)
        return z,theta

    #x = sqrt(4 m^2 - sqrt(4 m^4 + 8 m^2 p^2 - 3 m1^4 + 6 m1^2 m2^2 - 12 m1^2 p^2 z + 6 m1^2 p^2 - 3 m2^4 + 12 m2^2 p^2 z - 6 m2^2 p^2 - 12 p^4 z^2 + 12 p^4 z + p^4) - 3 m1^2 - 3 m2^2 - 6 p^2 z^2 + 6 p^2 z + p^2)/sqrt(6)
    def randz(self,mother,thetamin=1e-10,thetamax=0.4,isize=1):
       zmin=0.2/mother.mom.p,
       m=mother.mom.m/mother.mom.p
       mmax = np.minimum(mother.mom.m/mother.mom.p,0.4)
       z = beta.rvs(0.1,1, size=isize)
       theta=m/np.sqrt(z*(1-z))
       #dau2mass=self.dau2mass(mother,z,theta)
       count=0
       while(theta < thetamin or theta > thetamax or z < zmin or z > 0.5):
           z = beta.rvs(0.1,1, size=isize)
           theta=m/np.sqrt(z*(1-z))
           #dau2mass=self.dau2mass(mother,z,theta)
           count+=1
           if count > 1000:
            print("theta",m,z,theta,np.sqrt(z*(1-z)))
       return z,theta
    
    def mom_value(self,e,z1,t1,z2,t2,z):
        c=1+z*0.5*z1*t1**2+(1-z)*0.5*z2*t2**2
        return e/c
    
    def softsplit_try2(self, mother):
        np.random.seed()
        randomdraw_phi = 0#np.random.uniform(0,2*np.pi)
        randomdraw_theta=mother.randtheta
        zrand=mother.z
        if randomdraw_theta == -1000:
            zrand,randomdraw_theta=self.randz(zmin=0.2/mother.mom.p,m=mother.mom.m/mother.mom.p,isize=1)
        dau1_z,randtheta1 = self.randztheta(zmin=0.2/mother.mom.p,mmin=1e-10,mmax=mother.mom.m/mother.mom.p,isize=1)
        dau2_z,randtheta2 = self.randztheta(zmin=0.2/mother.mom.p,mmin=1e-10,mmax=mother.mom.m/mother.mom.p,isize=1)
        dau1_m = self.massp(dau1_z,randtheta1,mother.mom.p) #mass was dimensionless in sampler
        dau2_m = self.massp(dau2_z,randtheta2,mother.mom.p)
        #print("!!!",dau1_z,dau1_m,zrand,randomdraw_theta,dau2_m)
        dau1_phi =  randomdraw_phi
        dau2_phi =  randomdraw_phi + np.pi
        dau1_phi %= (2*np.pi)
        dau2_phi %= (2*np.pi)
        dau1_theta = randomdraw_theta*np.cos(dau1_phi)
        dau1_phi   = randomdraw_theta*np.sin(dau1_phi)
        dau2_theta = (zrand)/(1-zrand)*randomdraw_theta*np.cos(dau2_phi)
        dau2_phi   = (zrand)/(1-zrand)*randomdraw_theta*np.sin(dau2_phi)
        eta1   = self.theta_to_eta(dau1_theta+np.pi/2)
        eta2   = self.theta_to_eta(dau2_theta+np.pi/2)
        dau1_e = np.sqrt((dau_p*zrand)**2+dau1_m**2)
        dau2_e = np.sqrt((dau_p*(1-zrand))**2+dau2_m**2)
        dau1_mom = Momentum4.e_m_eta_phi(dau1_e[0], dau1_m[0], eta1[0], dau1_phi[0])
        dau2_mom = Momentum4.e_m_eta_phi(dau2_e[0], dau2_m[0], eta2[0], dau2_phi[0])
        dau1_mom = self.rotateTheta(dau1_mom,mother.mom.theta-np.pi/2)
        dau2_mom = self.rotateTheta(dau2_mom,mother.mom.theta-np.pi/2)
        dau1_mom = self.rotatePhi(dau1_mom,mother.mom.phi)
        dau2_mom = self.rotatePhi(dau2_mom,mother.mom.phi)
        dau2_mom=mother.mom-dau1_mom
        print("Test",dau2_mom.m,"-",dau2_m)
        #sumvec=dau1_mom+dau2_mom
        #print("test2",sumvec.p,sumvec.m,sumvec.eta,"--",mother.mom.p,mother.mom.m,mother.mom.eta)
        #print("test",sumvec.p,sumvec.m,sumvec.eta,"--",mother.mom.p,mother.mom.m,mother.mom.eta)
        #send it off
        dau1 = particle(mom=dau1_mom,randtheta=randtheta1,z=dau1_z)
        dau2 = particle(mom=dau2_mom,randtheta=randtheta2,z=dau2_z)
        self.zsoft.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.z.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.randtheta.append(randomdraw_theta)
        return dau1,dau2

    def checkm1m2m(self,m,m1,m2):
        v1=m**2-m1**2-m2**2
        v2=m1*np.sqrt(1+m2**2)
        return v1 > v2

    def fullform(self,z,theta):
        v1=z*(1-z)
        v2=theta**2
        v3=np.sqrt(z**2+theta**2)*np.sqrt((1-z)**2+theta**2)
        return np.sqrt(2*(v1+v2+v3))

    
    def theta_func(self,z,m,m1,m2,p):
        val0=(1./(4.*p**2))*(m**4-2*(m**2)*(m1**2+m2**2)+(m1**2-m2**2)**2)
        val1=z*(m1**2-m2**2)
        val2=m1**2
        val3=z*(1-z)*m**2
        num=(val0+val1-val2+val3)
        den=(p**2+m**2)
        return np.arctan(np.sqrt(num/den))

    def ptheta(self,z,m,m1,m2,p):
        first=4*m**2+p**2-3*m1**2-3*m2**2+6*z*(1-z)*p**2
        second=4*(m**2-2*p**2)*m**2-3*(m1**2-m2**2)**2+(6*p**2-12*z*p**2)*(m1**2-m2**2+2*z*p**2)+(12*z**2+1)*p**4
        return np.arctan(np.sqrt((first-np.sqrt(second))/6.))

    def dau2(self,iMother,iM1,iTheta,iZ,iPhi):
        dau1_m  = iM1
        dau1_px = iZ*iMother.mom.p
        dau1_pz = np.tan(iTheta)*iMother.mom.p
        dau1_e  = np.sqrt(dau1_px**2+dau1_pz**2+dau1_m**2)
        dau1_theta = iTheta*np.cos(iPhi)
        dau1_phi   = iTheta*np.sin(iPhi)
        dau1_eta   = self.theta_to_eta(-dau1_theta+np.pi/2)
        dau1_e = np.sqrt(dau1_px**2+dau1_pz**2+dau1_m**2)
        d1=Momentum4.e_m_eta_phi(dau1_e[0], dau1_m, dau1_eta[0],dau1_phi[0])
        mo=iMother.mom
        d1 = self.rotateTheta(d1,iMother.mom.theta-np.pi/2)
        d1 = self.rotatePhi  (d1,iMother.mom.phi)
        d2 = mo-d1
        return d1,d2

    def checkdau2(self,iMother,iM1,iTheta,iZ,iPhi):
        d1,d2=self.dau2(iMother,iM1,iTheta,iZ,iPhi)
        return np.iscomplex(d2.m)

    def randz(self,mother,iPhi,thetamin=1e-10,thetamax=0.4,isize=1):
        m=mother.mom.m
        p=mother.mom.p
        zmin=np.maximum(0.2/mother.mom.p,0.5*(1-np.sqrt(1-(m/p)**2)))
        zmax=0.5
        z  = beta.rvs(0.1,1, size=isize)
        z1 = beta.rvs(0.1,1, size=isize)
        t1 = beta.rvs(0.1,1, size=isize)
        z2 = beta.rvs(0.1,1, size=isize)
        t2 = beta.rvs(0.1,1, size=isize)
        m1 = self.fullform(z1,t1)*p*z
        m2 = self.fullform(z2,t2)*p*(1-z)
        theta=self.theta_func(z,m,m1,m2,p)
        count=0
        while(theta < thetamin or theta > thetamax or (np.isnan(theta)) or z < zmin or z > zmax
              or z1 > 0.5 or z > 0.5 or t1 > 0.5 or t2 > 0.5
              or m1 < 0.1 or m2 < 0.1 
              or (not self.checkm1m2m(m,m1,m2)) or self.checkdau2(mother,m1,theta,z,iPhi)
                ):
          z  = beta.rvs(0.1,1, size=isize)
          z1 = beta.rvs(0.1,1, size=isize)
          t1 = beta.rvs(0.1,1, size=isize)
          z2 = beta.rvs(0.1,1, size=isize)
          t2 = beta.rvs(0.1,1, size=isize)
          m1 = self.fullform(z1,t1)*p*z
          m2 = self.fullform(z2,t2)*p*(1-z)
          theta=self.theta_func(z,m,m1,m2,p)
          count+=1
          if count > 1000:
            #print("Sampled more than 1000 times")
            #print("theta",m,p,z,theta,m1,m2,self.checkm1m2m(m,m1,m2),self.checkdau2(mother,m1,theta,z,iPhi))
            return -1,-1,-1,-1
        return z,theta,m1,m2
   
    def softsplit(self, mother):
        np.random.seed()
        randomdraw_phi = np.random.uniform(0,2*np.pi)
        randomdraw_theta=mother.randtheta
        zrand=mother.z
        rand_m1=mother.m1
        rand_m2=mother.m2
        if randomdraw_theta == -1000:
            zrand,randomdraw_theta,rand_m1,rand_m2=self.randz(mother=mother,iPhi=randomdraw_phi,isize=1)
            mother.z=zrand
            mother.randtheta=randomdraw_theta
            if zrand == -1:
                dau1 = particle(mom=mother.mom,randtheta=-1000,z=-1,m1=-1000,m2=-1000)
                dau2 = particle(mom=mother.mom,randtheta=-1000,z=-1,m1=-1000,m2=-1000)
                
                #print("randomtheta: ", randomdraw_theta[0])
                #print(dau1.mom.p_t[0], dau2.mom.p_t[0])
                return dau1,dau2, -111.11, -111.11
        dau1_mom,dau2_mom=self.dau2(mother,rand_m1,randomdraw_theta,zrand,randomdraw_phi)
        dau1 = particle(mom=dau1_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)
        dau2 = particle(mom=dau2_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)
        #print("dau1 m",dau1.mom,"dau2 m",dau2.mom)
        self.z.append(np.min([dau1.mom.p_t[0], dau2.mom.p_t[0]])/(dau1.mom.p_t[0]+dau2.mom.p_t[0]))
        self.zsoft.append(np.min([dau1.mom.p_t[0], dau2.mom.p_t[0]])/(dau1.mom.p_t[0]+dau2.mom.p_t[0]))
        self.randtheta.append(randomdraw_theta[0])
        #print(dau1, dau2, np.min([dau1.mom.p_t[0], dau2.mom.p_t[0]])/(dau1.mom.p_t[0]+dau2.mom.p_t[0]), randomdraw_theta)
        #print("randomtheta: ", randomdraw_theta[0])
        return dau1, dau2, np.min([dau1.mom.p_t[0], dau2.mom.p_t[0]])/(dau1.mom.p_t[0]+dau2.mom.p_t[0]), randomdraw_theta[0]

    
    def hardsplit(self, mother, nthsplit):
        #Hard splitting performed in the rest frame of the mother, rotated, and then lorentz boosted back
        #Hard splitting prior: Gaussian around pi/2,
        np.random.seed()
        #randomdraw_theta = np.abs(np.random.normal(np.pi/2,0.1))
        randomdraw_theta = np.random.uniform(0.1,np.pi/2.-0.1)
        randomdraw_phi   = np.random.uniform(0,2*np.pi)
        if nthsplit==1:
            randomdraw_phi = 0
        #print("hard", nthsplit," ", mother.m)
        dau1_m = np.random.uniform(mother.mom.m/16, mother.mom.m/2)
        dau2_m = np.random.uniform(mother.mom.m/16, mother.mom.m/2)
        #if nthsplit == 1 and self.nprong > 2:
        #    dau1_m = 80.379
        #    dau2_m = 40.18
        #else:  
        #    dau1_m = 40.18
        #    dau2_m = 40.18
        if nthsplit == 1:
            dau1_m = 80.379
            dau2_m = 40.18
            
        if nthsplit == 2:
            dau1_m = 40.18
            dau2_m = 40.18
            
        dau1_theta = (np.pi/2 + randomdraw_theta)
        dau2_theta = (np.pi/2 - randomdraw_theta)        
        dau1_phi = mother.mom.phi + randomdraw_phi
        dau2_phi = mother.mom.phi + randomdraw_phi + np.pi
        dau1_phi %= (2*np.pi)
        dau2_phi %= (2*np.pi)
        #prep for 4-vector
        dau_p2   = self.p2(dau1_m,dau2_m,mother.mom.m)
        dau1_e   = np.sqrt(dau_p2+dau1_m**2)
        dau2_e   = np.sqrt(dau_p2+dau2_m**2)
        dau1_mom = Momentum4.e_m_eta_phi(dau1_e, dau1_m, self.theta_to_eta(dau1_theta), dau1_phi)
        dau2_mom = Momentum4.e_m_eta_phi(dau2_e, dau2_m, self.theta_to_eta(dau2_theta), dau2_phi)
        dau1_mom = self.rotateTheta(dau1_mom,mother.mom.theta-np.pi/2)
        dau2_mom = self.rotateTheta(dau2_mom,mother.mom.theta-np.pi/2)
        dau1_mom = self.rotatePhi(dau1_mom,mother.mom.phi)
        dau2_mom = self.rotatePhi(dau2_mom,mother.mom.phi)
        dau1 = particle(mom=dau1_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)
        dau2 = particle(mom=dau2_mom,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)
        dau1.mom = dau1.mom.boost_particle(mother.mom)
        dau2.mom = dau2.mom.boost_particle(mother.mom)
        self.randtheta.append(randomdraw_theta)
        self.zhard.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        self.z.append(np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t))
        return dau1, dau2, np.min([dau1.mom.p_t, dau2.mom.p_t])/(dau1.mom.p_t+dau2.mom.p_t), randomdraw_theta


    def draw_first_particle(self):
        #Draw mass from pdf
        np.random.seed()
        if self.massprior == "signal":
            m = np.random.normal(172.76, 1.32)
            if self.nprong == 4:
                m = np.random.normal(500, 1.32)
                

        if self.massprior == "background":
            m = np.random.uniform(0, 100)
        
        p = np.random.exponential(400)
        #delete later
        if self.doFixP:
            p = 400
        vec0 = Momentum4.m_eta_phi_p(m, 0, 0, p)
        part = particle(mom=vec0,randtheta=-1000,z=-1000,m1=-1000,m2=-1000)  
        return part

    def hard_decays(self):
        hardparticle_list = [self.draw_first_particle()]
        prong = 1
        zlist = []
        thetalist = []
        while prong < self.nprong:
            dau1, dau2, z, theta = self.hardsplit(hardparticle_list[0],prong)
            
            #print(dau1.mom.m, dau2.mom.m)
            hardparticle_list.pop(0)
            self.reverse_insort(hardparticle_list, dau1)
            self.reverse_insort(hardparticle_list, dau2)
            zlist.append(z)
            thetalist.append(theta)
            prong += 1
            
        return hardparticle_list, zlist, thetalist

    def genshower(self,_):
        showered_list, zlist, thetalist = self.hard_decays()
        total_particle = len(showered_list)        
        while total_particle < self.nparticle:
            if showered_list[0].mom.p < 1:
                break
            #print(self.softsplit(showered_list[0]))
            dau1, dau2, z, theta = self.softsplit(showered_list[0])
            if dau1.z == -1:
                break
            #print(dau1.mom,showered_list)
            showered_list.pop(0)
            self.reverse_insort(showered_list, dau1)
            self.reverse_insort(showered_list, dau2)
            
            zlist.append(z)
            thetalist.append(theta)
            
            total_particle +=1
        return total_particle, showered_list, zlist, thetalist
    
    def shower(self,_):
        i=0

        total_particle,showered_list,zlist, thetalist=self.genshower(i)
        #print(total_particle, self.nparticle)
        while total_particle <  self.nparticle:
            total_particle,showered_list,zlist, thetalist=self.genshower(i)
        arr = []

        check = Momentum4(0,0,0,0)
        for j in range(self.nparticle):
            #print("WITH NO INDEX",showered_list[j].mom.p_t, showered_list[j].mom.eta, showered_list[j].mom.phi )
            #print("WITH 0 INDEX",showered_list[j].mom.p_t[0], showered_list[j].mom.eta[0], showered_list[j].mom.phi[0] )
            arr.append(showered_list[j].mom.p_t)
            arr.append(showered_list[j].mom.eta)
            arr.append(showered_list[j].mom.phi)
            check += showered_list[j].mom


        #print("squeeze",np.squeeze(np.array(arr)).shape, np.squeeze(np.array(zlist)).shape, np.squeeze(np.array(thetalist)).shape)
        return np.squeeze(np.array(arr)), np.squeeze(np.array(zlist)), np.squeeze(np.array(thetalist))

    def generate_dataset(self, nevent):

        #output = torch.FloatTensor([])

        data = np.empty([nevent, 3*self.nparticle], dtype=float)
        data_z     = np.empty([nevent, self.nparticle-1], dtype=float)
        data_theta = np.empty([nevent, self.nparticle-1], dtype=float)
        
        if self.doMultiprocess:
            pool = Pool(processes=self.ncore)
            data, data_z, data_theta  = zip(*pool.map(self.shower,range(nevent)))

        else:
            for i in range(nevent):
                if i % 10 == 0:
                    print("event :",i)
                arr, arr_z, arr_theta = self.shower(i)    
                data[i]  = np.squeeze(arr)
                data_z[i] = np.squeeze(arr_z)
                data_theta[i] = np.squeeze(arr_theta)
        

        #return output
        return np.array(data), np.array(data_z), np.array(data_theta)


    #def visualize_one_event(self):


