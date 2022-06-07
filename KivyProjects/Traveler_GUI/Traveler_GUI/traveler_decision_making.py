'''
@author: Shipeng Liu/RoboLand
@feature: Suggest some proper locations to be measured base on current 
          measurements
'''

from re import T
from statistics import mean
from scipy.optimize import curve_fit
from scipy import signal
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import seaborn as sns
import math


'''generate random gaussian variable
Args:
    mean: the mean of gaussian variable
    scale: the amptitude of gaussian variable
    x:size of gaussian variable
Returns:
    gaussian variable
'''
def gauss(mean, scale, x=np.linspace(1,22,22), sigma=1):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))


'''the hypothesis model function
Args:
    x: input
    P1,P2,P3: parameter
Returns:
    output
'''
def model(x, P1, P2, P3):
    result = [0] * len(x)
    for i in range(len(x)):
        result[i] = P1 - P2 * max(P3 - x[i], 0)
    return result

'''the hypothesis model function
Args:
    xx: 
    yy: 
    zz:
    moist: 
Returns:
    err:
    xfit:
    model:
'''
def hypofit(xx, yy, zz):
    xx = np.array(inputs['xx'])
    yy = np.array(inputs['yy'])
    zz = np.array(inputs['zz'])
    
    P0 = [8, 0.842, 9.5]
    lb = [0, 0, 0]
    ub = [20, 5, 20]

    Pfit, covs = curve_fit(model, xx, yy, P0, bounds=(lb, ub))
    xfit = np.linspace(0, 16, 17)
    unique_x = np.unique(xx)
    loc = np.unique(zz)

    RMSE_average = [0] * len(unique_x)
    RMSE_spread = [0] * len(unique_x)

    for i in range(len(unique_x)):
        aa = np.nonzero(xx == unique_x[i])[0]
        xx_finded = xx[aa]
        yy_finded = yy[aa]
        RMSE_average[i] = (np.abs(np.mean(yy_finded) - 
                        np.mean(model(xx_finded, Pfit[0], Pfit[1], Pfit[2]))))
        RMSE_spread[i] = np.std(yy_finded, ddof=1)

    xx_model = model(xfit, Pfit[0], Pfit[1], Pfit[2])

    output = {
        'loc': loc.tolist(), 
        'err': RMSE_average, 
        'spread': RMSE_spread, 
        'xfit': xfit.tolist(), 
        'xx_model': xx_model,
        'Pfit': Pfit.tolist()
    }

    return output

#Find peaks API Route
'''the hypothesis model function
Args:
    spatial_reward: reward matrix considering spatial factor 
    moisture_reward: reward matrix considering variable factor
    discrepancy_reward: reward matrix considering discrepancy factor
Returns:
    suggested location which has maximum 
    spatial reward/moisture reward/discrepancy reward.
'''
def findPeaks(spatial_reward, moisture_reward, discrepancy_reward):

    disrepancy_reward_negative = np.array(discrepancy_reward) * -1

    spatial_locs, spatial_properties = signal.find_peaks(spatial_reward, 
                                                        height=0.3, distance=2)
    variable_locs, variable_properties = signal.find_peaks(moisture_reward,
                                                         height=0.3, distance=2)
    discrepancy_locs, discrepancy_properties = signal.find_peaks(
                                    discrepancy_reward, height=0.2, distance=2)
    discrepancy_lows_locs, discrepancy_lows_properties = signal.find_peaks(
                            disrepancy_reward_negative, height=-0.5, distance=2)

    max_used_spatial = False
    max_used_variable = False
    max_used_discrepancy = False
    max_used_discrepancy_lows = False

    if len(spatial_locs) == 0:
        spatial_locs = np.array([np.argmax(spatial_reward)])
        max_used_spatial = True
    
    if len(variable_locs) == 0:
        variable_locs = np.array([np.argmax(moisture_reward)])
        max_used_variable = True

    if len(discrepancy_locs) == 0:
        discrepancy_locs = np.array([np.argmax(discrepancy_reward)])
        max_used_discrepancy = True
        
    if len(discrepancy_lows_locs) == 0:
        discrepancy_lows_locs = np.array([np.argmax(disrepancy_reward_negative)])
        max_used_discrepancy_lows = True

    output = {
        'spatial_locs': spatial_locs.tolist(),
        'variable_locs': variable_locs.tolist(),
        'discrepancy_locs': discrepancy_locs.tolist(),
        'discrepancy_lows_locs': discrepancy_lows_locs.tolist(),
        'max_used_spatial': max_used_spatial,
        'max_used_variable': max_used_variable,
        'max_used_discrepancy': max_used_discrepancy,
        'max_used_discrepancy_lows': max_used_discrepancy_lows
    }

    return output





'''
@This class is the decision making module which is reponsible for processing the
 current state and compute four different rewards to suggest proper locations.
'''
class DecisionMaking:
    def __init__(self):
        '''Initial env info and parameters for decision making
        '''
        self.stage = ['Initial', 'Exploration', 'Verification']
        self.current_stage = 0
        self.spatial_information_coverage = []
        self.variable_information_coverage = []
        self.discrepancy_coverage = []
        self.current_confidence = 0
        self.beliefmatrix = []
        self.current_belief = 0
        self.current_state_location = []
        self.current_state_sample = []
        self.current_state_moisture = []
        self.current_state_shear_strength = []
        self.strategy = 0


    '''
    update the user's belief toward the hypothesis
    '''
    def update_belief(self, belief, confidence):
        self.current_belief = belief
        self.current_confidence = confidence

    '''
    update the data state and sync the measured data
    '''
    def update_current_state(self, location, sample, moisture, shear_strenth):
        sorted_index = np.argsort(location)
        self.current_state_location = location[sorted_index]
        self.current_state_sample = sample[sorted_index]
        self.current_state_moisture = moisture[sorted_index]
        self.current_state_shear_strength = shear_strenth[sorted_index]

    '''
    compute the spatial reward coverage 
    '''
    def handle_spatial_information_coverage(self):
        I_s = np.zeros(22)   #information matrix in location
        location = self.current_state_location
        sample = self.current_state_sample

        for jj in range(len(location)):
            I_s_s = np.exp(-1/np.sqrt(sample(jj)))
            I_s += gauss(location(jj) - 1, I_s_s, np.linspace(1,22,22), 1.5)
            I_s[I_s > 1] = 1

        self.spatial_information_coverage = I_s
        self.spatial_reward = 1 - I_s
        return self.spatial_reward

    '''
    compute the variable reward coverage
    '''
    def handle_variable_information_coverage(self):
        location = self.current_state_location
        moisture_bins = np.linspace(-1,17,19)
        xx = self.current_state_moisture.flatten()
        countMoist, bins, patches = plt.hist(xx, moisture_bins)
        I_v_s = np.exp(-1/np.sqrt(2*countMoist))

        I_v = np.zeros(len(I_v_s))
        for jj in range(len(I_v_s)):
            I_v += gauss(jj - 1, I_v_s(jj), moisture_bins, 1.5)
            I_v[I_v > 1] = 1
        self.variable_information_coverage = I_v

        mean_moisture_each = []
        min_moisture_each = []
        max_moisture_each = []
        for jj in range(len(location)):
            if(jj == 0):
                moisture = self.current_state_moisture[jj]
                moisture_mean = np.mean(moisture)
                moisture_std = np.std(moisture)
                ## find the next point b
                moisture_next = self.current_state_moisture[jj+1]
                moisture_mean_next = np.mean(moisture_next)
                moisture_std_next = np.std(moisture_next)
                ## compute the slope of ab
                slope = ((moisture_mean_next - moisture_mean) 
                                / (location[jj+1] - location[jj]))
                for loc in range(location[jj]):
                    mean_moisture_each[loc] = moisture_mean - slope * ( 
                                                            location[jj] - loc)
                    min_moisture_each[loc] =  math.min(moisture_mean - 
                                2 * slope * (location[jj] - loc), moisture_mean)  
                    max_moisture_each[loc] = math.max(moisture_mean - 
                                2 * slope * (location[jj] - loc), moisture_mean)
            elif(jj == len(location) - 1):
                moisture = self.current_state_moisture[jj]
                moisture_mean = np.mean(moisture)
                moisture_std = np.std(moisture)
                moisture_prev = self.current_state_moisture[jj-1] 
                moisture_mean_prev = np.mean(moisture_prev)
                moisture_std_prev = np.std(moisture_prev)
                slope = (moisture_mean - moisture_mean_prev) / (location[jj]
                             - location[jj-1])
                for loc in range(location[jj], 21):
                    mean_moisture_each[loc] = moisture_mean + slope * (
                                                           loc - location[jj])
                    min_moisture_each[loc] =  math.min(moisture_mean + 
                                2 * slope * (22 - location[jj]), moisture_mean)  
                    max_moisture_each[loc] = math.max(moisture_mean +
                                2 * slope * (22 - location[jj]), moisture_mean)   
                for loc in range(location[jj-1], location[jj]):
                    mean_moisture_each[loc] = moisture_mean + slope * (loc - 
                                        location[jj])
                    min_moisture_each[loc] = math.min(moisture_mean_prev, 
                                        moisture_mean) 
                    max_moisture_each[loc] = math.max(moisture_mean_prev,
                                        moisture_mean) 
            else:
                moisture = self.current_state_moisture[jj]
                moisture_mean = np.mean(moisture)
                moisture_std = np.std(moisture)
                moisture_prev = self.current_state_moisture[jj-1] 
                moisture_mean_prev = np.mean(moisture_prev)
                moisture_std_prev = np.std(moisture_prev)
                slope = (moisture_mean - moisture_mean_prev) / (location[jj]
                             - location[jj-1])
                for loc in range(location[jj-1], location[jj]):
                    mean_moisture_each[loc] = moisture_mean + slope * (loc - 
                                        location[jj])
                    min_moisture_each[loc] = math.min(moisture_mean_prev, 
                                        moisture_mean) 
                    max_moisture_each[loc] = math.max(moisture_mean_prev,
                                        moisture_mean) 
        R_v_set = np.zeros(22)
        self.mean_moisture_each = mean_moisture_each
        self.min_moisture_each = min_moisture_each
        self.max_moisture_each = max_moisture_each
        for jj in range(22):
            std = (max_moisture_each[jj] - min_moisture_each[jj])/3
            moisture_possibility = np.linspace(min_moisture_each(jj), 
                                                max_moisture_each(jj), 10)
            probability = gauss(mean_moisture_each[jj], 1, moisture_possibility,
                                                std)
            actual_probability = probability/np.sum(probability)
            R_m_l = 0
            for ii in range(len(moisture_possibility)):
                if(round(moisture_possibility[ii]) + 2 < 1):
                    moisture_index = 0
                elif(round(moisture_possibility[ii]) + 2 > 17):
                    moisture_index = 16
                else:
                    moisture_index = round(moisture_possibility[ii]) + 2
                R_m_l = R_m_l + I_v[moisture_index] * actual_probability[ii]
            R_v_set[jj] = R_m_l
        self.variable_reward = 1 - R_v_set
        return self.variable_reward

    '''
    compute discrepancy reward coverage
    '''
    def handle_discrepancy_coverage(self):
        MinCoverage = 0.06
        moisture_bins = np.linspace(-1,17,19)
        xx = self.current_state_moisture.flatten()
        yy = self.current_state_shear_strength.flatten()
        zz = []
        for jj in range(len(self.current_state_location)):
            zz.append(self.current_state_location[jj] * np.ones((1, 
                                                self.current_state_sample[jj])))
        RMSE = []
        location = self.current_state_location
        sort_index = np.argsort(xx)
        xx_sorted = xx[sort_index]
        yy_sorted = yy[sort_index]
        zz_sorted = zz[sort_index]
        countMoist, bins, patches = plt.hist(xx, moisture_bins)
        moistcoverage = len(np.nonzero(countMoist))/len(moisture_bins)
        if(moistcoverage > MinCoverage):
            loc, RMSE_distribution, RMSE_spread, xfit, xx_model = hypofit(
                xx_sorted, yy_sorted, zz_sorted
            )
        else:
            xx_unique = np.unique(xx_sorted)
            RMSE_distribution = 0.5 * np.ones(len(xx_unique))    

        ## compute the belief of shearstrenght vs moisture
        xx_unique = np.unique(xx_sorted)
        xx_mean = []
        yy_mean = []
        for i in range(len(xx_unique)):
            aa = np.argwhere(xx_sorted==xx_unique[i])
            xx_finded = xx_sorted[aa]
            yy_finded = yy_sorted[aa]
            xx_mean[i] = np.mean(xx_finded)
            yy_mean[i] = np.mean(yy_finded)
        shearstrength_predict = []
        shearstrength_min = []
        shearstrength_max = []
        for i in range(len(xx_mean)):
            if(i == 1):
                moisture_mean = xx_mean[i]
                shearstrength_mean = yy_mean[i]
                moisture_mean_prev = xx_mean[i-1]
                shearstrength_mean_prev = yy_mean[i]
                slope = (shearstrength_mean - shearstrength_mean_prev)/(
                                    moisture_mean - moisture_mean_prev)
                for jj in range(np.ceil(xx_unique[i])+2 , 19):
                    shearstrength_predict[jj] = shearstrength_mean + slope * (
                          jj - moisture_mean -2
                      )
                    shearstrength_min[jj] = math.min(2 * 
            shearstrength_predict[-1] - shearstrength_mean,  shearstrength_mean)
                    shearstrength_max[jj] = math.max(2 * 
            shearstrength_predict[-1] - shearstrength_mean,  shearstrength_mean)
                for jj in range(np.ceil(xx_unique[i-1])+2, 
                                                np.floor(xx_unique[i])+2):
                    shearstrength_predict[jj] = shearstrength_mean + slope * (
                          jj - moisture_mean - 2)
                    shearstrength_min[jj] = math.min(yy_mean[i-1], yy_mean[i])
                    shearstrength_max[jj] = math.max(yy_mean[i-1], yy_mean[i])
            else:
                moisture_mean = xx_mean[i]
                shearstrength_mean = yy_mean[i]
                moisture_mean_prev = xx_mean[i-1]
                shearstrength_mean_prev = yy_mean[i]
                slope = (shearstrength_mean - shearstrength_mean_prev)/(
                                    moisture_mean - moisture_mean_prev)
                for jj in range(np.ceil(xx_unique[i-1])+2, 
                                                np.floor(xx_unique[i])+2):
                    shearstrength_predict[jj] = shearstrength_mean + slope * (
                          jj - moisture_mean - 2)
                    shearstrength_min[jj] = math.min(yy_mean[i-1], yy_mean[i])
                    shearstrength_max[jj] = math.max(yy_mean[i-1], yy_mean[i])
        shearstrength_range = shearstrength_max - shearstrength_min

        # compute the potential discrepancy reward
        R_d_set = np.zeros((22))
        for jj in range(22):
            std_moist = (self.max_moisture_each[jj] - self.min_moisture_each[jj])/3
            moisture_possibility = np.linspace(self.min_moisture_each(jj), 
                                                self.max_moisture_each(jj), 10)
            probability = gauss(self.mean_moisture_each[jj], 1, 
                                moisture_possibility, std_moist)
            moisture_actual_probability = probability/np.sum(probability)
            R_d_m = 0
            for kk in range(len(moisture_possibility)):
                if(np.round(moisture_possibility[kk]) + 2 < 1):
                    moisture_index = 1
                elif(np.round(moisture_possibility[kk]) + 2 > 17):
                    moisture_index = 17
                else:
                    moisture_index = np.round(moisture_possibility[kk]) + 2
                shearstrength_std = shearstrength_range(moisture_index)/3
                shearstrength_possibility = np.linspace(
                                            shearstrength_min[moisture_index],
                                            shearstrength_max[moisture_index],
                                            10)
                shearstrength_probability = gauss(
                                        shearstrength_predict[moisture_index],  
                                1, shearstrength_possibility, shearstrength_std)
                shearstrength_actual_prob = (shearstrength_probability/
                                            np.sum(shearstrength_probability)) 
                shearstrength_hypo_value = xx_model(moisture_index)
                R_d_l = 0
                for qq in range(len(shearstrength_possibility)):
                    R_d_l = (R_d_l +shearstrength_actual_prob[qq] * 
                                np.abs(shearstrength_possibility[qq] -
                                             shearstrength_hypo_value))
                R_d_m = R_d_m + R_d_l * moisture_actual_probability[kk]
            R_d_set[jj] = R_d_m
        self.discrepancy_coverage = R_d_set/3
        self.discrepancy_reward = R_d_set/3

    # give the final suggested location choice and then pass it to user
    # user interface 
    def calculate_suggested_location(self):
        output = findPeaks(self.spatial_reward, self.variable_reward, 
                                            self.discrepancy_reward)
        return output
        



if __name__ == '__main__':
    Traveler_DM = DecisionMaking()
    
    Traveler_DM.update_current_state()
    Traveler_DM.handle_spatial_information_coverage()
    Traveler_DM.handle_variable_information_coverage()
    Traveler_DM.handle_discrepancy_coverage()
    Traveler_DM.calculate_suggested_location()
    