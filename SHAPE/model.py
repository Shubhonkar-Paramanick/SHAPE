#!/usr/bin/env python3
import numpy as np
import scipy as sp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import math
import emcee
import corner
import pickle
import tkinter as tk
import matplotlib.font_manager
from tkinter import simpledialog
from tkinter.simpledialog import askinteger
import easygui
from IPython.display import display, Math
from numpy.linalg import inv
import time
from time import perf_counter
from tqdm import tqdm
from traits.api import HasTraits, Instance, Array, \
    on_trait_change
from traitsui.api import View, Item, HGroup, Group
from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from pylab import *
import datetime
from mpl_toolkits.basemap import Basemap
import os,imageio


#np.set_printoptions(suppress=True)
plt.rcParams['figure.figsize'] = [12,7]
plt.rcParams["font.family"] = "Times New Roman"


Data_List = pd.read_excel('ERBE_Data_Nov_1986.xlsx')
Data = np.array(Data_List)
Lat_Data = Data[:,0]
Lat_Data = Lat_Data[np.logical_not(np.isnan(Lat_Data))]
Lon_Data = Data[0,:]
Lon_Data = Lon_Data[~np.isnan(Lon_Data)]

Albedo_Grid = np.delete(np.delete(Data, 0,0), 0,1)/100.

A_Lat = []
for i in range(0,len(Lat_Data)):
    a = np.average(Albedo_Grid[i,:])
    A_Lat.append(a)
Avg_Alb_Lat = np.stack(A_Lat, axis=0)
plt.plot(Lat_Data, Avg_Alb_Lat,'c-')
plt.xlabel('Latitude')
plt.ylabel('Albedo') 
plt.title('') 
plt.grid(True) 
plt.show()


# Range of the variation in Percentage

Lat_Range = (max(Avg_Alb_Lat)-min(Avg_Alb_Lat))*100./(min(Avg_Alb_Lat))




A_Lon = []
for j in range(0,len(Lon_Data)):
    b = np.average(Albedo_Grid[:,j])
    A_Lon.append(b)
Avg_Alb_Lon = np.stack(A_Lon, axis=0)
plt.plot(Lon_Data, Avg_Alb_Lon,'k-')
plt.xlabel('Longitude')
plt.ylabel('Albedo') 
plt.title('') 
plt.grid(True) 
plt.show()


# Range of the variation in Percentage

Lon_Range = (max(Avg_Alb_Lon)-min(Avg_Alb_Lon))*100./(min(Avg_Alb_Lon))



# Albedo Model (Latitude Only)




Alb_Err = 0.02*np.mean(Avg_Alb_Lat)*np.random.rand(len(Lat_Data))


plt.errorbar(Lat_Data, Avg_Alb_Lat, Alb_Err, color='black', fmt='o', capsize=5, capthick=1, ecolor='black')
plt.xlabel('Latitude')
plt.ylabel('Albedo') 
plt.grid()
plt.show(block=False)

Lat_Model = np.linspace(min(Lat_Data),max(Lat_Data),10000)


Lat_Avg_Data = (Lat_Data,Avg_Alb_Lat,Alb_Err)







############################################################################################
############################## Model (y0 + A0*cos²(θ+θ₀)) ################################
############################################################################################
def Albmodel_ml(Parameters,Theta=Lat_Model):
    y0, A0, Theta0 = Parameters
    return y0 + (A0*((np.cos((np.pi*Theta/180.)+Theta0))**2))



#
# Likelihood Function
#
def lnlike(Parameters,Theta,y,yerror):
    return -0.5*(np.sum((np.log(2*np.pi*(yerror**2)))+(((y-Albmodel_ml(Parameters,Theta))/yerror)**2)))



#
# Initial Parameters, Lower and Upper Bounds
#
Param0 = np.array([0.4, -0.6, 3.0])
bounds = ((-3.0,3.0),(-3.0,3.0),(0.0,2*np.pi))



#
# Maximum Likelihood Fit
#
ln = lambda*args: -lnlike(*args)
output = minimize(ln, Param0, args=Lat_Avg_Data, bounds=bounds, tol= 1e-9)
offset_ml, amplitude_ml, phase_ml = output["x"]
ml_estimates = np.array([offset_ml, amplitude_ml, phase_ml])
ndim = len(ml_estimates)
best_fit_model_ml = Albmodel_ml(ml_estimates,Lat_Model)



plt.errorbar(Lat_Data, Avg_Alb_Lat, Alb_Err, color='black', fmt='o', capsize=5, capthick=1, ecolor='black')
plt.plot(Lat_Model,best_fit_model_ml,color='#064b8a', label="Maximum Likelihood Fit")
plt.xlabel('Latitude')
plt.ylabel('Albedo')
plt.legend(fontsize=14)
plt.grid()
plt.show()

from sympy import symbols, diff, pi, cos, sin
yi, y_0, a0, theta, theta_0, d_yi = symbols('yi y_0 a0 theta theta_0 d_yi', real=True)
func = (-1/2)*(((yi-(y_0 + (a0*((cos((pi*theta/180.)+theta_0))**2))))/(d_yi))**2)

pd_y_0 = diff(func, y_0)
pd2_y_0 = diff(pd_y_0, y_0)

pd_a0 = diff(func, a0)
pd2_a0 = diff(pd_a0, a0)

pd_theta_0 = diff(func, theta_0)
pd2_theta_0 = diff(pd_theta_0, theta_0)


a = 1.0/Alb_Err**2
b = np.divide(1.0*np.cos(0.00555555555555556*np.pi*Lat_Data + phase_ml)**4,Alb_Err**2)
c = np.divide((4.0*amplitude_ml**2*np.sin(0.00555555555555556*np.pi*Lat_Data + phase_ml)**2*np.cos(0.00555555555555556*np.pi*Lat_Data + phase_ml)**2) - (2.0*amplitude_ml*(-amplitude_ml*np.cos(0.00555555555555556*np.pi*Lat_Data + phase_ml)**2 - offset_ml + Avg_Alb_Lat)*np.sin(0.00555555555555556*np.pi*Lat_Data + phase_ml)**2) + (2.0*amplitude_ml*(-amplitude_ml*np.cos(0.00555555555555556*np.pi*Lat_Data + phase_ml)**2 - offset_ml + Avg_Alb_Lat)*np.cos(0.00555555555555556*np.pi*Lat_Data + phase_ml)**2),Alb_Err**2)

Unct_y0 = np.sqrt(1/(np.sum(a)))
Unct_A0 = np.sqrt(1/(np.sum(b)))
Unct_Theta0 = np.sqrt(1/(np.sum(c)))
Uncertainties = np.array([Unct_y0, Unct_A0, Unct_Theta0],dtype = ('float64'))

print('\033[1m' + 'Parameters using Maximum Likelihood Method (Minimizing the negative Log-Likelihood)' + '\033[0m')
label = ['y₀', 'A₀', 'θ₀']
for s in range(ndim):
    txt = "\mathrm{{{2}}} = {0:.9f}_{{-{1:.3f}}}^{{{1:.3f}}}"
    txt = txt.format(ml_estimates[s], Uncertainties[s], label[s])
    display(Math(txt))







#create array to cover parameter space

from itertools import product
xi, yi, zi = np.ogrid[-100.0:100.0:100j, -100.0:100.0:100j, -6.28:6.28:50j]

m = xi.flatten()
n = yi.flatten()
o = zi.flatten()
mno = np.asarray(list(product(m, n, o)))


#calculate likelihoods

t0 = perf_counter();
L = np.array([lnlike(mno[v],Lat_Data,Avg_Alb_Lat,Alb_Err) for v in tqdm(range(len(mno)))])
t1 = perf_counter();
Execution_Time = t1-t0




#t0 = time.time();
#results = []
#for i in tqdm(range(len(mno))):
#    results.append(lnlike(mno[i],x_data,Rad_Vel,RV_Error))
#LUP = np.hstack(results)
#t1 = time.time();
#Execution_Time = t1-t0


#select parameter with maximum likelihood
Likelihood_max = np.max(L)
MaxPos = mno[np.array([p for p,q in tqdm(enumerate(L)) if q==np.max(L)])]
#draw likelihood

  
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
img = ax.scatter(mno[:,0], mno[:,1], mno[:,2], c=L, lw=0, s=20, cmap=plt.viridis())
ax.set_xlabel('$y₀$', fontsize=17)
ax.set_ylabel('$A₀$', fontsize=17)
ax.set_zlabel('$θ₀$', fontsize=17)
fig.colorbar(img)
plt.show()





#Volume Slice in Mayavi

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
                                MlabSceneModel
################################################################################

L_New = L.reshape(len(m),len(n),len(o))

################################################################################
# The object implementing the dialog
class VolumeSlicer(HasTraits):
    # The data to plot
    L_New = Array()

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    L_New_src3d = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    _axis_names = dict(x=0, y=1, z=2)


    #---------------------------------------------------------------------------
    def __init__(self, **traits):
        super(VolumeSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z


    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _L_New_src3d_default(self):
        return mlab.pipeline.scalar_field(self.L_New,
                            figure=self.scene3d.mayavi_scene)

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.L_New_src3d,
                        figure=self.scene3d.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name)
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')


    #---------------------------------------------------------------------------
    # Scene activation callbaks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        outline = mlab.pipeline.outline(self.L_New_src3d,
                        figure=self.scene3d.mayavi_scene,
                        )
        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            # Turn the interaction off
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()


    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)

        # To avoid copying the data, we take a reference to the
        # raw VTK dataset, and pass it on to mlab. Mlab will create
        # a Mayavi source from the VTK without copying it.
        # We have to specify the figure so that the data gets
        # added on the figure we are interested in.
        outline = mlab.pipeline.outline(
                            self.L_New_src3d.mlab_source.dataset,
                            figure=scene.mayavi_scene,
                            )
        ipw = mlab.pipeline.image_plane_widget(
                            outline,
                            plane_orientation='%s_axes' % axis_name)
        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Synchronize positions between the corresponding image plane
        # widgets on different views.
        ipw.ipw.sync_trait('slice_position',
                            getattr(self, 'ipw_3d_%s'% axis_name).ipw)

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0
        # Add a callback on the image plane widget interaction to
        # move the others
        def move_view(obj, evt):
            position = obj.GetCurrentCursorPosition()
            for other_axis, axis_number in self._axis_names.items():
                if other_axis == axis_name:
                    continue
                ipw3d = getattr(self, 'ipw_3d_%s' % other_axis)
                ipw3d.ipw.slice_position = position[axis_number]

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.L_New.shape[
                    self._axis_names[axis_name]]

        # Position the view for the scene
        views = dict(x=( 0, 90),
                     y=(90, 90),
                     z=( 0,  0),
                     )
        scene.mlab.view(*views[axis_name])
        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)


    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')


    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(HGroup(
                  Group(
                       Item('scene_y',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene_z',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene_x',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene3d',
                            editor=SceneEditor(scene_class=MayaviScene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                ),
                resizable=True,
                title='Volume Slicer',
                )


vs_plot = VolumeSlicer(L_New=L_New)
vs_plot.configure_traits()







############################################################################################
############################## Model (y0 + A0*cos²(θω₀+θ₀)) ################################
############################################################################################
def Albmodel_ml(Parameters,Theta=Lat_Model):
    y0, A0, Omega0, Theta0 = Parameters
    return y0 + (A0*((np.cos((Omega0*np.pi*Theta/180.)+Theta0))**2))



#
# Likelihood Function
#
def lnlike(Parameters,Theta,y,yerror):
    return -0.5*(np.sum((np.log(2*np.pi*(yerror**2)))+(((y-Albmodel_ml(Parameters,Theta))/yerror)**2)))



#
# Initial Parameters, Lower and Upper Bounds
#
Param0 = np.array([0.4, -0.6, 0.5, 3.0])
bounds = ((-3.0,3.0),(-3.0,3.0),(0.0,1.5),(0.0,2*np.pi))



#
# Maximum Likelihood Fit
#
ln = lambda*args: -lnlike(*args)
output = minimize(ln, Param0, args=Lat_Avg_Data, bounds=bounds, tol= 1e-9)
offset_ml, amplitude_ml, ang_freq_ml, phase_ml = output["x"]
ml_estimates = np.array([offset_ml, amplitude_ml, ang_freq_ml, phase_ml])
ndim = len(ml_estimates)
best_fit_model_ml = Albmodel_ml(ml_estimates,Lat_Model)



plt.errorbar(Lat_Data, Avg_Alb_Lat, Alb_Err, color='black', fmt='o', capsize=5, capthick=1, ecolor='black')
plt.plot(Lat_Model,best_fit_model_ml,color='#008080', label="Maximum Likelihood Fit")
plt.xlabel('Latitude')
plt.ylabel('Albedo')
plt.legend(fontsize=14)
plt.grid()
plt.show()




yi, y_0, a0, omega_0, theta, theta_0, d_yi = symbols('yi y_0 a0 omega_0 theta theta_0 d_yi', real=True)
func = (-1/2)*(((yi-(y_0 + (a0*((cos((pi*omega_0*theta/180.)+theta_0))**2))))/(d_yi))**2)

pd_y_0 = diff(func, y_0)
pd2_y_0 = diff(pd_y_0, y_0)

pd_a0 = diff(func, a0)
pd2_a0 = diff(pd_a0, a0)

pd_omega_0 = diff(func, omega_0)
pd2_omega_0 = diff(pd_omega_0, omega_0)

pd_theta_0 = diff(func, theta_0)
pd2_theta_0 = diff(pd_theta_0, theta_0)




a = 1.0/Alb_Err**2
b = np.divide(1.0*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**4,Alb_Err**2)
c = np.divide((0.000123456790123457*np.pi**2*amplitude_ml**2*Lat_Data**2*np.sin(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2) - (6.17283950617284e-5*np.pi**2*amplitude_ml*Lat_Data**2*(-amplitude_ml*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2 - offset_ml + Avg_Alb_Lat)*np.sin(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2) + (6.17283950617284e-5*np.pi**2*amplitude_ml*Lat_Data**2*(-amplitude_ml*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2 - offset_ml + Avg_Alb_Lat)*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2),Alb_Err**2)
d = np.divide((4.0*amplitude_ml**2*np.sin(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2) - (2.0*amplitude_ml*(-amplitude_ml*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2 - offset_ml + Avg_Alb_Lat)*np.sin(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2) + (2.0*amplitude_ml*(-amplitude_ml*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2 - offset_ml + Avg_Alb_Lat)*np.cos(0.00555555555555556*np.pi*ang_freq_ml*Lat_Data + phase_ml)**2),Alb_Err**2)



Unct_y0 = np.sqrt(1/(np.sum(a)))
Unct_A0 = np.sqrt(1/(np.sum(b)))
Unct_Omega0 = np.sqrt(1/(np.sum(c)))
Unct_Theta0 = np.sqrt(1/(np.sum(d)))
Uncertainties = np.array([Unct_y0, Unct_A0, Unct_Omega0, Unct_Theta0],dtype = ('float64'))

print('\033[1m' + 'Parameters using Maximum Likelihood Method (Minimizing the negative Log-Likelihood)' + '\033[0m')
label = ['y₀', 'A₀', 'ω₀', 'θ₀']
for s in range(ndim):
    txt = "\mathrm{{{2}}} = {0:.9f}_{{-{1:.3f}}}^{{{1:.3f}}}"
    txt = txt.format(ml_estimates[s], Uncertainties[s], label[s])
    display(Math(txt))










################################################################################################
########################################## ERBE Data ###########################################
################################################################################################



###################### Density Plot with Basemap #####################

#Cylindrical

Lon_Data2, Lat_Data2 = np.meshgrid(Lon_Data,Lat_Data)

m = Basemap(projection='cyl',llcrnrlat=-90.,urcrnrlat=90.,\
            llcrnrlon=0.,urcrnrlon=360.,resolution='i')
x, y = m(Lon_Data2, Lat_Data2)

m.drawcoastlines()
m.drawparallels(np.arange(-90.,90.,10.0))
m.drawmeridians(np.arange(0.,360.,10.0))
m.drawmapboundary(fill_color='white')
m.drawcountries()
cs = m.contourf(x,y,Albedo_Grid,200, cmap=plt.cm.bwr)
#cs = m.contourf(x,y,Albedo_Grid,200, cmap=plt.cm.Spectral_r)
plt.title('Earth Albedo')
cbar = plt.colorbar(cs, orientation='horizontal', shrink=0.5)
plt.show()


#Ortho

Lon_Data2, Lat_Data2 = np.meshgrid(Lon_Data,Lat_Data)

m = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='i')
x, y = m(Lon_Data2, Lat_Data2)

m.drawcoastlines()
m.drawparallels(np.arange(-90.,90.,10.0))
m.drawmeridians(np.arange(0.,360.,10.0))
m.drawmapboundary(fill_color='white')
m.drawcountries()
c = m.contourf(x,y,Albedo_Grid,200, cmap=plt.cm.bwr)
#cs = m.contourf(x,y,Albedo_Grid,200, cmap=plt.cm.Spectral_r)
plt.title('Earth Albedo')
cbar = plt.colorbar(c, orientation='horizontal', shrink=0.5)
plt.show()





#############################################################
###################### Save Images ##########################
#############################################################


def save_img(img_dir,image_indx,dpi=180):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # save each .png for GIF
    plt.savefig(img_dir+'image_'+str(image_indx)+'_.png',dpi=dpi)
    plt.close('all')





plt.ioff()

lat_viewing_angle = [-10.,-10.]
lon_viewing_angle = [-180.0,180.0]
rotation_steps = 150
lat_vec = np.linspace(lat_viewing_angle[0],lat_viewing_angle[0],rotation_steps)
lon_vec = np.linspace(lon_viewing_angle[0],lon_viewing_angle[1],rotation_steps)



image_indx = 0
min_alb = Albedo_Grid.min()
max_alb = Albedo_Grid.max()
# loop through the longitude vector above
for pp in tqdm(range(0,len(lat_vec))):    
    plt.cla()
    m = Basemap(projection='ortho', 
              lat_0=lat_vec[pp], lon_0=lon_vec[pp])
    
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines()
    m.drawcountries()
    Lon_Data2, Lat_Data2 = np.meshgrid(np.append(Lon_Data, Lon_Data[-1]+Lon_Data[1]-Lon_Data[0]),Lat_Data)
    x, y = m(Lon_Data2, Lat_Data2)
    m.contourf(x,y,np.hstack((Albedo_Grid, Albedo_Grid[:,[0]])),200, cmap=plt.cm.Spectral_r, levels = np.linspace(min_alb,max_alb,200)), plt.colorbar(orientation='horizontal', shrink=0.5)
    save_img('./Images/',image_indx,dpi=180)
    image_indx+=1




#############################################################
######################## Make GIF ###########################
#############################################################


img_dir = './Images/'

images,image_file_names = [],[]
for file_name in os.listdir(img_dir):
    if file_name.endswith('.png'):
        image_file_names.append(file_name)       
sorted_files = sorted(image_file_names, key=lambda y: int(y.split('_')[1]))
frame_length = 0.1 # Frames duration
end_pause = 0.1 # Last frame duration
for i in range(0,len(sorted_files)):       
    file_path = os.path.join(img_dir, sorted_files[i])
    if i==len(sorted_files)-1:
        for j in range(0,int(end_pause/frame_length)):
            images.append(imageio.imread(file_path))
    else:
        images.append(imageio.imread(file_path))
imageio.mimsave('ERBE_Data.gif', images,'GIF',duration=frame_length)








################################################################################################
##################### Spherical Harmonic Analysis of the Albedo Function #######################
################################################################################################



N_0=15
N_1=36.


def A(m):
    return [Albedo_Grid[:,[j]]*np.cos(m*Lon_Data[j]*np.pi/180.) for j in range(0,len(Lon_Data))]

A_m = []
for m in range(0,N_0+1):
    A_m.append(A(m))

A_m = np.einsum('mno->nom', np.sum(np.array(A_m), axis=1))



def B(m):
    return [Albedo_Grid[:,[j]]*np.sin(m*Lon_Data[j]*np.pi/180.) for j in range(0,len(Lon_Data))]

B_m = []
for m in range(0,N_0+1):
    B_m.append(B(m))

B_m = np.einsum('mno->nom', np.sum(np.array(B_m), axis=1))




def d(i):
    return [((2**0.5)/N_0)*(np.sin((i+7.5)*np.pi/(2*N_1)))*np.sum(np.fromiter(((1./((2.0*h)+1.))*(np.sin(((2.0*h)+1.)*((i+7.5)*np.pi/(2*N_1)))) for h in range(0,N_0)), float))]


d_i = []
for i in range(0,len(Lat_Data)):
    d_i.append(d(i))

d_i = np.asarray(d_i).flatten()



# Legendre polynomial
def P(n, m, theta):
    if(n == 0 and n == m):
        return theta/theta
    if(n != 0 and n == m):
        return ((((2*n)+1)/(2*n))**0.5)*(np.sin(np.pi*theta/180.))*(P(n-1, n-1, theta))
    elif(n>=m+2):
        return ((((((2*n)-1)*((2*n)+1))/((n-m)*(n+m)))**0.5)*(np.cos(np.pi*theta/180.))*(P(n-1, m, theta)))-((((((2*n)+1)*(n+m-1)*(n-m-1))/(((2*n)-3)*(n+m)*(n-m)))**0.5)*(P(n-2, m, theta)))
    elif(m<n<m+2):
        return (((2*n)+1)**0.5)*(np.cos(np.pi*theta/180.))*(P(n-1, m, theta))
    else:
        return print("Error!")



P_nm = []
for n in range(0,N_0+1):
    for m in range(0,n+1):
        P_nm.append(P(n, m, theta=Lat_Data))


A_nm = []

for n in range(0,N_0+1):
    for m in range(0,n+1):
        A_nm.append((np.pi/N_0)*np.sum(P(n, m, theta=Lat_Data)*d_i*A_m[:,:,m].flatten(),axis=0))



B_nm = []

for n in range(0,N_0+1):
    for m in range(0,n+1):
        B_nm.append((np.pi/N_0)*np.sum(P(n, m, theta=Lat_Data)*d_i*B_m[:,:,m].flatten(),axis=0))



THETA = np.array([float(x) for x in np.linspace(min(Lat_Data),max(Lat_Data),30) if x != 0])
PHI = np.linspace(min(Lon_Data),max(Lon_Data),30)


#THETA = np.array([float(x) for x in np.linspace(-90.,90.,100) if x != 0])
#PHI = np.linspace(0.,360.,100)


def Albedo(theta, phi):
    ret = 0
    for n in range(0, N_0+1):
        for m in range(0, n+1):
            function = (((np.pi/N_0)*np.sum(P(n, m, theta=Lat_Data)*d_i*A_m[:,:,m].flatten(),axis=0))*(math.cos(m*phi*np.pi/180.))*(P(n, m, theta))) + (((np.pi/N_0)*np.sum(P(n, m, theta=Lat_Data)*d_i*B_m[:,:,m].flatten(),axis=0))*(math.sin(m*phi*np.pi/180.))*(P(n, m, theta)))
            ret += function
    return ret


Albedo_Grid_SH = np.array([[Albedo(theta, phi) for phi in PHI]for theta in tqdm(THETA)])

Albedo_Grid_SH = Albedo_Grid_SH/np.max(Albedo_Grid_SH)






###############################################################################################################

#Cylindrical

PHI_SH, THETA_SH = np.meshgrid(PHI,THETA)

m_sh = Basemap(projection='cyl',llcrnrlat=-90.,urcrnrlat=90.,\
            llcrnrlon=0.,urcrnrlon=360.,resolution='i')
x_sh, y_sh = m_sh(PHI_SH, THETA_SH )

m_sh.drawcoastlines()
m_sh.drawparallels(np.arange(-90.,90.,10.0))
m_sh.drawmeridians(np.arange(0.,360.,10.0))
m_sh.drawmapboundary(fill_color='white')
m_sh.drawcountries()
cs_sh = m_sh.contourf(x_sh, y_sh,Albedo_Grid_SH,200, cmap=plt.cm.bwr)
#cs_sh = m.contourf(x_sh, y_sh,Albedo_Grid_SH,200, cmap=plt.cm.Spectral_r)
plt.title('Earth Albedo')
cbar_sh = plt.colorbar(cs_sh, orientation='horizontal', shrink=0.5)
plt.show()


#Ortho

PHI_SH, THETA_SH = np.meshgrid(PHI,THETA)

m_sh = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='i')
x_sh, y_sh = m_sh(PHI_SH, THETA_SH )

m_sh.drawcoastlines()
m_sh.drawparallels(np.arange(-90.,90.,10.0))
m_sh.drawmeridians(np.arange(0.,360.,10.0))
m_sh.drawmapboundary(fill_color='white')
m_sh.drawcountries()
cs_sh = m_sh.contourf(x_sh, y_sh,Albedo_Grid_SH,200, cmap=plt.cm.bwr)
#cs_sh = m.contourf(x_sh, y_sh,Albedo_Grid_SH,200, cmap=plt.cm.Spectral_r)
plt.title('Earth Albedo')
cbar_sh = plt.colorbar(cs_sh, orientation='horizontal', shrink=0.5)
plt.show()






















###############################################################################################


