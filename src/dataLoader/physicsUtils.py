import math
import numpy as np
from scipy.constants import G

# Mass of each planet in kg as defined by IAU in 2009
mass = {
    'earth' : 5.9722e24,
    'mercury' : 3.3010e23, 	
    'venus' : 4.1380e24,
    'mars' : 6.4273e23,
    'jupiter' : 1.89852e27,
    'saturn' : 5.6846e26,
    'uranus' : 8.6819e25,
    'neptune' : 1.02431e26
}

mass_sun = 1.989e30

# Average orbital distance in m
radius = { 
    'neptune' :	4.49506e12,
    'uranus' :	2.87246e12,
    'saturn' :	1.43353e12,
    'jupiter' :	7.7857e11,
    'mars' :	2.2792e11,
    'earth' :	1.496e11,
    'venus' :	1.0821e11,
    'mercury' :	5.791e10
}

# Energy of planets (potential, kinetic, total) in joules
energy = {
    'mercury' : (-7.5526e32, 	3.79059e32, 	-3.76201e32),
    'venus' : 	(-5.98571e33, 	2.9749e33,  	-3.01081e33),
    'earth' : 	(-5.29201e33, 	2.66762e33, 	-2.62439e33),
    'mars' : 	(-3.73775e32, 	1.86979e32, 	-1.86796e32),
    'jupiter' : (-3.24179e35, 	1.62046e35, 	-1.62133e35),
    'saturn' : 	(-5.28556e34, 	2.65372e34, 	-2.63184e34),
    'uranus' : 	(-4.01788e33, 	2.00961e33, 	-2.00827e33),
    'neptune' : (-3.02974e33, 	1.5216e33,  	-1.50814e33)
}

def getEnergyTotal(normalize=False):
    if normalize:
        return None
    else:
        return [val[2] for val in energy]


# This function assumes perfect eccentricity
# @returns tuple representing velocity_x, velocity_y in m/s
def getAvgVelocity(x, y):
    muSun = 1.32712440018e20
    radius = np.sqrt(x**2 + y**2)
    mag = np.sqrt(muSun/radius)
    theta = np.arctan2(y, x) + (math.pi / 2.0)

    return mag * np.cos(theta), mag * np.sin(theta)

def getVelocity(semiMajorAxis, x, y):
    muSun = 1.32712440018e20
    radius = np.sqrt(x**2 + y**2)
    magsqr = muSun * ((2.0/radius) - (1.0/semiMajorAxis))
    mag = np.sqrt(np.clip(magsqr,0,None))
    theta = np.arctan2(y, x) + (math.pi / 2.0)

    return mag * np.cos(theta), mag * np.sin(theta)
