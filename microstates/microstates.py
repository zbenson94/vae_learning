import numpy as np
import os

import granular.dataloading as ssio
# ------------------------------------------------------------------------------------
#    This function prepares the inputs for the VAE. They are the pressure 
#    on the compression wall. The defaults are given for the smaller
#    system that I simulated. I assume the labels are the number of cycles
# ------------------------------------------------------------------------------------
#### AUTHOR: ZAB
#### DATE:   4 May 2021
# ------------------------------------------------------------------------------------
def prepInputMEMS(hdr,frames,dimx=5.0,dimz=10.0,x0=0.075,radius=0.25,N=5000):

    dg     = np.sqrt(2)*radius
    Nx     = int(np.ceil(dimx/dg))
    Nz     = int(np.ceil(dimz/dg))
    binx   = dg * np.arange(Nx)
    binz   = dg * np.arange(Nz)

    Nms    = int(1000 * len(frames))
    output = np.zeros((Nms,Nx,Nz,1),dtype='float32')
    labels = np.zeros(Nms,dtype='int')

    cntr   = 0 
    # ------------------------------------------------------------------------------------
    # Independent trials
    # ------------------------------------------------------------------------------------
    trials = os.listdir('{}/'.format(hdr))
    for trial in trials:
        if(trial.startswith('trial')):
            # ----------------------------------------------------------------------------
            # Cycles for each trial
            # ----------------------------------------------------------------------------
            cycles = os.listdir('{}/{}'.format(hdr,trial))
            for cycle in cycles:
                # ------------------------------------------------------------------------
                # Frames I want for the VAE input
                # ------------------------------------------------------------------------
                for frame in frames:
                    if(cycle.endswith('cycle{}'.format(frame))):
                        fname = '{}/{}/{}/run2/ss.0000100'.format(hdr,trial,cycle)
                        if(os.path.isfile(fname)):
                            xyz = ssio.read_SS(fname)[0][0:N,:]
                            # Get the wall contacts
                            idw  = (xyz[:,1] - x0) < radius
                            xyzc = xyz[idw,:]

                            idx  = np.digitize(xyzc[:,0],binx) - 1
                            idz  = np.digitize(xyzc[:,2],binz) - 1

                            output[cntr,idx,idz,0] = 1000*(radius - (xyzc[:,1] - x0))
                            labels[cntr]           = frame
                            cntr = cntr + 1
                            
    # In case some files are missing
    output = output[0:cntr,:]
    labels = labels[0:cntr]
    
    return output,labels