import numpy as np
import os
import sys
import scipy
import scipy.io
import minkowski as mn
import bandpass as bp
import pandas as pd
from mpi4py import MPI
# import psutil

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

##########################################################################
#---Parameters to specify domain division to distribute among ranks------#
ndivx=8 # Number of divisions in x-direction
ndivz=4 # Number of divisions in z-direction
##########################################################################
#---Data Parameters - Grid size, number of points and Reynolds number----#
[nx, ny, nz, Lx, Ly, Lz, Re_tau, Re]=bp.grid()
##########################################################################
# Load velocity data from binary files
# S=30 # Filter length scale
if len(sys.argv) > 1:
    S = int(sys.argv[1])
    print(f"Ruuning minkowski code for S={S}")
else:
    raise ValueError("Please provide the filter length scale as input in the command-line argument.")
##########################################################################
##########################################################################
fpath1="./results/scaling/" # Path to the raw data files
y=1-np.cos(np.arange((ny//2)+1)*np.pi/(ny-1)) # Chebyshev y-coordinate
# y=np.empty(ny)
# with open('ymesh4000.dat','rb') as fid2:
#     y=np.fromfile(fid2,'>f')              # y-coordinate for the finite-difference schemes
yp=y[0:(ny//2+1)]*Re_tau
x=np.arange(nx)*(Lx/nx)
z=np.arange(nz)*(Lz/nz)

##########################################################################
#-Three zones for Minkowski analysis are divided equally among all ranks-#
# Determine subdomain
num_groups = 3
group_id = rank // (size // num_groups)  # assign group number
# Validate processor count
assert size == num_groups*ndivx*ndivz, f"This code requires exactly {num_groups*ndivx*ndivz} MPI ranks"

# Create a sub-communicator for each group
sub_comm = comm.Split(color=group_id, key=rank)

sub_rank = sub_comm.Get_rank()
sub_size = sub_comm.Get_size()

# Global subdomain y-ranges
ind1=np.argmax(yp>30)
ind2=np.argmax(yp>200)
y_ranges = [(0, ind1), (ind1, ind2), (ind2, (ny//2)+1)]
y_start, y_end = y_ranges[group_id]
local_ny = y_end - y_start

# Determine split block size
local_nx = nx // ndivx
local_nz = nz // ndivz
local_shape = (local_nx, local_ny, local_nz)

# # Allocate receive buffer
local_block = np.empty(local_shape, dtype=np.float32)

##############################################################################

for i in range(num_groups):
    if group_id==i and sub_rank==0:
        fname=fpath1+"ens"+str(S)
        with open(fname,"rb") as fid:
            data=np.fromfile(fid,'float32')
        ens=np.reshape(data,(nx,ny,nz),order='F')
        y_part = ens[:, y_start:y_end, :]
        del ens
        thres = np.mean(y_part)+5*np.std(y_part)  # threshold to determine iso-surfaces
        print(f"Mean:{thres}")
    comm.Barrier()

if sub_rank == 0:    
    all_blocks = np.empty((sub_size, local_nx, local_ny, local_nz), dtype=np.float32)
   
    # Fill all_blocks
    block_idx = 0
    x_blocks = np.split(y_part, ndivx, axis=0)
    for xb in x_blocks:
        z_blocks = np.split(xb, ndivz, axis=2)
        for zb in z_blocks:
            all_blocks[block_idx] = zb
            block_idx += 1
else:
    all_blocks = None
    thres = None


thres=sub_comm.bcast(thres, root=0)
sub_comm.Scatter(all_blocks, local_block, root=0)

comm.Barrier()
zone=group_id+1
fpath2="./minkowski4/var_lf"+str(S)+"/zone"+str(zone) # path where the output is written
fname2="Rank"+str(sub_rank) # folder name
xloc=np.split(x, ndivx)[sub_rank%ndivx] # local x-local coordinate for each rank
zloc=np.split(z, ndivz)[sub_rank//ndivx] # local z-local coordinate for each rank
yloc=y[y_start:y_end] # local y-local coordinate for each rank
fpath3=os.path.join(fpath2, "output.csv")
strmax = 3 # Maximum numer of structures to be detected by each rank
lmin = 5 # Minimum expected size of the structure along any direction
a = (1/Re_tau)*(zone==1)+(yloc[1]-yloc[0])*(zone!=1) # size of the cubic lattice to be used for interpolation
output=mn.pick(local_block,local_nx,local_ny,local_nz,thres,
               strmax,fname2,fpath2,lmin,zone,a,xloc,yloc,zloc,sub_rank)


gathered = sub_comm.gather(output, root=0)

if sub_rank == 0:
    # Convert to a single NumPy array
    gathered_array = np.concatenate(gathered, axis=0)
    df=pd.DataFrame(gathered_array)
    df.columns=["Count","V0","V1","V2","V3","MT","MW","ML","P","F","dmin","dmid","dmax"]
    df.to_csv(fpath3)

MPI.Finalize()


