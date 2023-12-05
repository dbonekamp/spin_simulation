import sys
import numpy as np
import cmath
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class Vector:
    def __init__(self, x, y, z):
        self.value = np.array([x,y,z])
        self.precision_digits=8
    @classmethod
    def fromVector(cls, vect):
        return cls(vect.value[0], vect.value[1], vect.value[2])           
    def rotatez(self, angle):
        selfcopy = Vector.fromVector(self)
        rotmatz = np.array([[    math.cos(angle), -math.sin(angle), 0.0],
                                [math.sin(angle), math.cos(angle), 0.0],
                                [0.0, 0.0, 1.0]
                        ])
        selfcopy.value = np.dot(rotmatz, selfcopy.value)
        return selfcopy
    def rotatex(self, angle):
        selfcopy = Vector.fromVector(self)
        rotmatz = np.array([   [1.0, 0.0, 0.0],
                            [  0.0, math.cos(angle), -math.sin(angle)],
                                [0.0, math.sin(angle), math.cos(angle)],
                                
                        ])
        selfcopy.value = np.dot(rotmatz, selfcopy.value)
        return selfcopy    
    def print(self):
        print(self.toString())
    def toString(self):
        return f"({self.value[0]}, {self.value[1]}, {self.value[2]})"
    def mult(self, factor):
        selfcopy = Vector.fromVector(self)
        selfcopy.value = selfcopy.value * factor
        return selfcopy
    def add(self, summand):
        selfcopy = Vector.fromVector(self)
        selfcopy.value = selfcopy.value + summand.value
        return selfcopy   
    def norm(self):
        return np.linalg.norm(self.value) 
    def xy_z(self):
        return [self.projectxy().norm(), self.zComponent()] # norm not needed for z-Component 
    def round(self):
        selfcopy = Vector.fromVector(self)
        selfcopy.value = np.around(selfcopy.value, decimals=selfcopy.precision_digits) 
        return selfcopy
    def projectxy(self):
        return Vector(self.value[0], self.value[1], 0.0)
    def zComponent(self):
        return self.value[2]
    def negative(self):
        selfcopy = Vector.fromVector(self)
        selfcopy.value = -selfcopy.value
        return selfcopy   
    
class Spin:
    def __init__(self, frequency, magnitude):
        self.frequency = frequency
        self.magnitude = magnitude
        self.vector = Vector(0.0, 0.0, 1.0).mult(magnitude);
        self.relaxation_rate = 0.02 
    def getFrequency(self):
        return self.frequency
    def precess(self, time_msec):
        angle = ((time_msec/1000)*2.0*math.pi)*self.frequency
        #print(f"angle = {angle}")
        self.vector = self.vector.rotatez(angle)
    def toString(self):
        return f"Frequency = {self.frequency}, vector = {self.vector.toString()}, magnitude = {self.magnitude}"
    def print(self):
        print(self.toString())        
    def magnitude(self):
        return np.norm(self.vector.projectxy())    
    def reset(self):
        self.vector = Vector(0.0, 0.0, self.vector.norm())  
    def rotatex(self, angle):
        self.vector = self.vector.rotatex(angle)
    def relax(self, milliseconds):
        relaxed_z = Vector(0,0,1).mult(self.magnitude) #relaxation goes to the baseline magnitude-size z-Vector
        relaxdiff = relaxed_z.add(self.vector.negative())
        self.vector = self.vector.add(relaxdiff.mult(self.relaxation_rate * milliseconds))

class Spinensemble:
    def __init__(self, number_spins, offset_frequency, stdev_offset):
        magnitude = 2.0
        self.spin_array = [ Spin(freq, magnitude) for freq in np.random.normal(offset_frequency, stdev_offset, number_spins)]
    def precess(self, time_msec):
        for spin in self.spin_array:
            spin.precess(time_msec)
    def relax(self, time_msec):
        for spin in self.spin_array:
            spin.relax(time_msec)        
    def magnitude_and_magnetization(self):
        res = Vector(0.0,0.0,0.0)
        for spin in self.spin_array:
            res=res.add(spin.vector)
        return [res.norm()] + res.xy_z()
    def magnitudeMagnetizationAfterPrecession(self,time_msec):     
        self.precess(time_msec)      
        return self.magnitude_and_magnetization
    def invert(self):
        for spin in self.spin_array:
            #spin.frequency = -spin.frequency
            spin.rotatex(np.deg2rad(180))  
    def phase_reset(self):
        for spin in self.spin_array:
            spin.reset()
    def x_pulse(self, angle_in_deg):
        for spin in self.spin_array:
            spin.rotatex(np.deg2rad(angle_in_deg))   

class Pulse:
    def __init__(self, type, time, value):
        self.time = time
        self.value = value
        self.type = type
        if self.type == 'x-pulse':
            self.color="m"
        elif self.type == 'invert':
            self.color="g"   
    def apply(self, ensemble):
        print(f"applying {self.toString()}")
        if self.type == 'x-pulse':
            ensemble.x_pulse(self.value)
        elif self.type == 'invert':
            ensemble.invert()
    def toString(self):
        if self.value:
            return f"{self.type}({self.time}, {self.value})"
        else:
            return f"{self.type}({self.time})"
    def expectedPeaks(self, peaks): #peak is [time, absolute height, relative height]
        expected = []
        for peak in peaks:
            TE_half = self.time - peak[0]
            # TE = 2 * TE_half
            absolute_time_of_echo = self.time + TE_half
            # absolute_time_of_echo = peak[0] + TE #gives same value
            expected.append([absolute_time_of_echo, None, None, self.toString() + "(" + peak[3] + ")"]) # cannot determine signal as not necessarily available yet
        return expected
    
class PulseSequence:
    def __init__(self):
        self.sequence = []
        self.expected_peaks = []
        self.peaks = []
        self.timepoints = None
        self.ensemble = None
        self.signal_over_time = []
    def add(self, pulse):
        self.sequence.append(pulse)
    def simulate(self, ensemble, timepoints):
        current_peak_label_asc = ord('A')
        self.timepoints = timepoints
        self.ensemble = ensemble
        ensemble.phase_reset()
        self.signal_over_time = []
        self.peaks = []
        self.expected_peaks = []
        prev_timepoint = timepoints[0]
        self.signal_over_time.append(ensemble.magnitude_and_magnetization())
        last_min = 0
        detect_thresh = 600
        cur_max = [None, 0, None]
        for i in tqdm(timepoints[1:]):
            elapsed = i - prev_timepoint
            ensemble.precess(elapsed)
            ensemble.relax(elapsed)
            timepoints_passed = [*range(prev_timepoint+1, i+1)]
            self.checkPulseAndApplyIfAppropriate(ensemble, timepoints_passed)
            #print(f"{timepoints_passed=}")
            magMag = ensemble.magnitude_and_magnetization()
            self.signal_over_time.append(magMag)
            mag = magMag[1] #0: for total Magnetization Norm, 1: for xy-plane magnetization (observable)
            if mag < last_min:
                last_min = mag
            if (mag - last_min > detect_thresh):
                if mag >= cur_max[1]:
                    cur_max = [i, mag, mag-last_min, chr(current_peak_label_asc)]
            if cur_max[0]:
                if (cur_max[1] - mag > detect_thresh):
                    self.peaks.append(cur_max)
                    cur_max=[None, 0, None]
                    last_min = mag
                    current_peak_label_asc+=1
            prev_timepoint = i
            self.sortExpectedPeaks()
        return [timepoints, self.signal_over_time, self.peaks]
    def checkPulseAndApplyIfAppropriate(self, ensemble, timepoints):
        for pulse in self.sequence:
            if pulse.time in timepoints:
                print(f"{pulse.time=}, {timepoints=}")
                #print(f"checking {pulse.toString()}")
                pulse.apply(ensemble)
                self.expected_peaks = self.expected_peaks + pulse.expectedPeaks(self.peaks)
    def plotWithSignal(self, signal):
        mag_signal = [a[0] for a in signal[1]]
        max_mag_signal = max(mag_signal)
        xy_signal = [a[1] for a in signal[1]]
        z_signal = [a[2] for a in signal[1]]
        print(mag_signal)
        for pulse in self.sequence:
            label = f"{pulse.toString()}"
            plt.axvline(x = pulse.time, color = pulse.color, label = label)
            plt.text(pulse.time, max_mag_signal*.8, label,rotation=90)
        plt.plot(signal[0], mag_signal, color = 'b', label='Magnitude')
        plt.plot(signal[0], xy_signal, color = 'c', label = 'xy Magnetization')
        plt.plot(signal[0], z_signal, color = 'g', label = 'z-Magnetization')
        plt.legend()
    def sortExpectedPeaks(self):
        self.expected_peaks = sorted(self.expected_peaks, key=lambda x: x[0])
    def determineHeightOfExpectedPeaks(self):
        prev_timepoint = self.timepoints[0]
        for i, j in zip(self.timepoints[1:], range(1,len(self.timepoints))):
            elapsed = i - prev_timepoint
            timepoints_passed = [*range(prev_timepoint+1, i+1)]  
            for k,l in zip(self.expected_peaks, range(0, len(self.expected_peaks))):
                #print(f"{k=}")
                if k[0] in timepoints_passed:
                    print(f"{k[0]} {j=} {self.signal_over_time[j][1]} {timepoints_passed=}")
                    cur_mag = self.signal_over_time[j][1]
                    self.expected_peaks[l][1] = cur_mag
            prev_timepoint = i
    def showExpectedPeaks(self):
        circle_radius = 5
        ydist = 50
        for p in self.expected_peaks:
            # circle = plt.Circle((p[0], p[1]), circle_radius, color = 'r')
            # plt.gca().add_patch(circle)
            plt.plot(p[0], p[1],'ro') 
            label = p[3]
            plt.text(p[0], p[1] + ydist, label,rotation=90)
    def showPeakLabels(self):
        circle_radius = 5
        ydist = 350
        xdist = 20
        for p in self.peaks:
            plt.plot(p[0], p[1],'go', markersize=10) 
            label = p[3]
            plt.text(p[0]-xdist, p[1] + ydist, label,rotation=0, fontsize=12)  
            label = f"({p[0]} msec)"
            plt.text(p[0]-xdist, p[1] + ydist*2, label,rotation=90, fontsize=8)       

