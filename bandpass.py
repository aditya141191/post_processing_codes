import os
import time
import numpy as np
import scipy.fft
from mpi4py import MPI

#--------------------------------------------------------------------------------------------------------------------------------------------#
def grid():
    with open('parameters.txt') as f:
        lines = f.readlines()

    nx=int(lines[0].split("\t")[1])
    ny=int(lines[1].split("\t")[1])
    nz=int(lines[2].split("\t")[1])

    if lines[3].split("\t")[2].replace("\n","")=="pi":
        Lx=float(lines[3].split("\t")[1])*np.pi
    else:
        Lx=float(lines[3].split("\t")[1])*float(lines[3].split("\t")[2])

    if lines[4].split("\t")[2].replace("\n","")=="pi":
        Ly=float(lines[4].split("\t")[1])*np.pi
    else:
        Ly=float(lines[4].split("\t")[1])*float(lines[4].split("\t")[2])

    if lines[5].split("\t")[2].replace("\n","")=="pi":
        Lz=float(lines[5].split("\t")[1])*np.pi
    else:
        Lz=float(lines[5].split("\t")[1])*float(lines[5].split("\t")[2])

    Re_tau=float(lines[6].split("\t")[1])
    Re=float(lines[7].split("\t")[1])

    return [nx, ny, nz, Lx, Ly, Lz, Re_tau, Re]
#--------------------------------------------------------------------------------------------------------------------------------------------#

def domain_info(comm):
    p=comm.Get_size()
    ny=grid()[1]
    nz=grid()[2]
    ysz=ny//p
    rem_y=ny%p

    zsz=nz//p
    rem_z=nz%p
    dom_info=[ysz, rem_y, zsz, rem_z]
    return dom_info
#--------------------------------------------------------------------------------------------------------------------------------------------#

def readfile(fname, comm):
    nx=grid()[0]
    ny=grid()[1]
    nz=grid()[2]

    my_rank=comm.Get_rank()
    p=comm.Get_size()
    [ysz,rem_y]=domain_info(comm)[0:2]

    if my_rank==0:
        with open(fname,"rb") as fid:
            data=np.fromfile(fid,'float32')
        data=np.reshape(data,(nx,ny,nz),order='F')
    else:
        data=None
    
    if rem_y==0:
        out=np.empty([nx,ysz,nz], dtype='float32')
        if my_rank==0:
            out=data[:,my_rank*(ysz):(my_rank+1)*(ysz),:]
            for rank in np.arange(1,p):
                buf=np.ascontiguousarray(np.copy(data[:,(rank*ysz):((rank+1)*ysz),:]))
                comm.Send(buf, dest=rank, tag=rank)
        else:
            comm.Recv(out, source=0, tag=my_rank)
    else:
        if my_rank in np.arange(rem_y):
            out=np.empty([nx,ysz+1,nz], dtype='float32')
        else:
            out=np.empty([nx,ysz,nz], dtype='float32')

        if my_rank==0:
            out=data[:,my_rank*(ysz+1):(my_rank+1)*(ysz+1),:]
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(data[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank)
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.ascontiguousarray(np.copy(data[:,rank*(ysz+1):(rank+1)*(ysz+1),:]))
                    comm.Send(buf, dest=rank, tag=rank)
                
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(data[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank)
        else:
            comm.Recv(out, source=0, tag=my_rank)

    return out

#--------------------------------------------------------------------------------------------------------------------------------------------#
def readfile64(fname, comm):
    nx=grid()[0]
    ny=grid()[1]
    nz=grid()[2]

    my_rank=comm.Get_rank()
    p=comm.Get_size()
    [ysz,rem_y]=domain_info(comm)[0:2]

    if my_rank==0:
        with open(fname,"rb") as fid:
            data=np.fromfile(fid,'float64')
        data=np.reshape(data,(nx,ny,nz),order='F')
    else:
        data=None
    
    if rem_y==0:
        out=np.empty([nx,ysz,nz], dtype='float64')
        if my_rank==0:
            out=data[:,my_rank*(ysz):(my_rank+1)*(ysz),:]
            for rank in np.arange(1,p):
                buf=np.ascontiguousarray(np.copy(data[:,(rank*ysz):((rank+1)*ysz),:]))
                comm.Send(buf, dest=rank, tag=rank)
        else:
            comm.Recv(out, source=0, tag=my_rank)
    else:
        if my_rank in np.arange(rem_y):
            out=np.empty([nx,ysz+1,nz], dtype='float64')
        else:
            out=np.empty([nx,ysz,nz], dtype='float64')
        
        if my_rank==0:
            out=data[:,my_rank*(ysz+1):(my_rank+1)*(ysz+1),:]
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(data[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank)
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.ascontiguousarray(np.copy(data[:,rank*(ysz+1):(rank+1)*(ysz+1),:]))
                    comm.Send(buf, dest=rank, tag=rank)
                
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(data[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank)
        else:
            comm.Recv(out, source=0, tag=my_rank)

    return out
#--------------------------------------------------------------------------------------------------------------------------------------------#

def writefile(fname, comm, input):
    nx=grid()[0]
    ny=grid()[1]
    nz=grid()[2]
    my_rank=comm.Get_rank()
    
    
    p=comm.Get_size()
    [nx,ny,nz]=grid()[0:3]
    [ysz,rem_y]=domain_info(comm)[0:2]
    
    if rem_y==0:
        if my_rank!=0:
            out=None
            comm.Send(input, dest=0, tag=my_rank)
        else:
            out=np.empty([nx,ny,nz], dtype=input.dtype)
            out[:,(my_rank*(ysz)):((my_rank+1)*(ysz)),:]=input
            for rank in np.arange(1,p):
                buf=np.empty([nx,ysz,nz], dtype=input.dtype)
                comm.Recv(buf, source=rank, tag=rank)
                out[:,(rank*ysz):((rank+1)*ysz),:]=buf 
    else:
        if my_rank!=0:
            out=None
            comm.Send(input, dest=0, tag=my_rank)
        else:
            out=np.empty([nx,ny,nz], dtype=input.dtype)
            out[:,(my_rank*(ysz+1)):((my_rank+1)*(ysz+1)),:]=input
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.empty([nx,ysz,nz], dtype=input.dtype)
                    comm.Recv(buf, source=rank, tag=rank)
                    out[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]=buf
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.empty([nx,ysz+1,nz], dtype=input.dtype)
                    comm.Recv(buf, source=rank, tag=rank)
                    out[:,(rank*(ysz+1)):((rank+1)*(ysz+1)),:]=buf
                
                for rank in np.arange(rem_y,p):
                    buf=np.empty([nx,ysz,nz], dtype=input.dtype)
                    comm.Recv(buf, source=rank, tag=rank)
                    out[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]=buf  

    if my_rank==0:
        with open(fname,"wb") as fid:
            out=np.reshape(out,(nx*ny*nz), order='F')
            fid.write(out.tobytes())

#--------------------------------------------------------------------------------------------------------------------------------------------#

def writePDF(fname, PDF, comm):
    ny=grid()[1]

    ysz=domain_info(comm)[0]
    rem_y=domain_info(comm)[1]

    my_rank=comm.Get_rank()
    p=comm.Get_size()

    (a,b,c)=np.shape(PDF)

    if rem_y==0:
        if my_rank!=0:
            comm.Send(PDF,dest=0,tag=my_rank)
        else:
            out=np.empty([a,b,ny], dtype=(PDF.dtype))
            out[:,:,(my_rank*(ysz)):((my_rank+1)*(ysz))]=PDF
            for rank in np.arange(1,p):
                buf=np.empty([a,b,ysz],dtype=(PDF.dtype))
                comm.Recv(buf, source=rank, tag=rank)
                out[:,:,(rank*(ysz)):((rank+1)*(ysz))]=buf
        
            out=np.reshape(out,(a*b*ny), order='F')
            np.savetxt(fname,out, delimiter="   ")
    else:
        if my_rank!=0:
            comm.Send(PDF,dest=0,tag=my_rank)
        else:
            out=np.empty([a,b,ny], dtype=(PDF.dtype))
            out[:,:,(my_rank*(ysz+1)):((my_rank+1)*(ysz+1))]=PDF
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.empty([a,b,ysz],dtype=(PDF.dtype))
                    comm.Recv(buf, source=rank, tag=rank)
                    out[:,:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y)]=buf
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.empty([a,b,ysz+1],dtype=(PDF.dtype))
                    comm.Recv(buf, source=rank, tag=rank)
                    out[:,:,(rank*(ysz+1)):((rank+1)*(ysz+1))]=buf

                for rank in np.arange(rem_y,p):
                    buf=np.empty([a,b,ysz],dtype=(PDF.dtype))
                    comm.Recv(buf, source=rank, tag=rank)
                    out[:,:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y)]=buf
        
            out=np.reshape(out,(a*b*ny), order='F')
            np.savetxt(fname,out, delimiter="   ")

#--------------------------------------------------------------------------------------------------------------------------------------------#
def reflect(u,y):
    (a,b,c)=np.shape(u)
    if b==len(y):
        u2=np.empty([a,(2*b-2),c], dtype=np.csingle)
        y2=np.empty([(2*b-2)], dtype=np.float32)
        u2[:,(b-2):(2*b-2),:]=u
        y2[(b-2):(2*b-2)]=y
        for n in np.arange((b-2)):
            u2[:,n,:]=-u[:,((b-3)-n+1),:]
            y2[n]=-y[-((b-3)-n+1)]
    else:
        print('Error: number of y-points not consistent in u and y')
        exit()
    return [u2, y2]
#--------------------------------------------------------------------------------------------------------------------------------------------#
def interpy(y,u,y2):
    nx=grid()[0]
    ny=grid()[1]

    N=len(y2)

    temp=np.empty(ny, dtype=np.csingle)
    c=np.shape(u)[2]
    u_inp=np.empty([nx,N,c], dtype=np.csingle)
    for i in np.arange(nx):
        for k in np.arange(c):
            temp=u[i,:,k]
            u_inp[i,:,k]=np.interp(y2,y,temp)

    del temp
    return u_inp
#--------------------------------------------------------------------------------------------------------------------------------------------#
def filter(u, L, y, comm):
    my_rank=comm.Get_rank()
    p=comm.Get_size()

    [nx, ny, nz, Lx, Ly, Lz, Re_tau, Re]=grid()
    [ysz, rem_y, zsz, rem_z]=domain_info(comm)
    
    
    L=L/Re_tau
    
    uk_loc=scipy.fft.fft2(u, s=[nx, nz], axes=(0,2))
    # del u
    uk_loc=uk_loc.astype(np.csingle)
    uk_loc_r=np.ascontiguousarray(np.copy(np.real(uk_loc)))
    uk_loc_i=np.ascontiguousarray(np.copy(np.imag(uk_loc)))
    kx=scipy.fft.fftfreq(nx, Lx/nx)*2*np.pi
    kz=scipy.fft.fftfreq(nz, Lz/nz)*2*np.pi
    comm.Barrier()

    #########################################################
    if rem_y==0:
        if my_rank!=0:
            ukr=None
            comm.Send(uk_loc_r, dest=0, tag=my_rank)
        else:
            ukr=np.empty([nx,ny,nz], dtype=uk_loc_r.dtype)
            ukr[:,(my_rank*(ysz)):((my_rank+1)*(ysz)),:]=uk_loc_r
            for rank in np.arange(1,p):
                buf=np.empty([nx,ysz,nz], dtype=uk_loc_r.dtype)
                comm.Recv(buf, source=rank, tag=rank)
                ukr[:,(rank*ysz):((rank+1)*ysz),:]=buf 
    else:
        if my_rank!=0:
            ukr=None
            comm.Send(uk_loc_r, dest=0, tag=my_rank)
        else:
            ukr=np.empty([nx,ny,nz], dtype=uk_loc_r.dtype)
            ukr[:,(my_rank*(ysz+1)):((my_rank+1)*(ysz+1)),:]=uk_loc_r
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.empty([nx,ysz,nz], dtype=uk_loc_r.dtype)
                    comm.Recv(buf, source=rank, tag=rank)
                    ukr[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]=buf 
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.empty([nx,ysz+1,nz], dtype=uk_loc_r.dtype)
                    comm.Recv(buf, source=rank, tag=rank)
                    ukr[:,(rank*(ysz+1)):((rank+1)*(ysz+1)),:]=buf
                
                for rank in np.arange(rem_y,p):
                    buf=np.empty([nx,ysz,nz], dtype=uk_loc_r.dtype)
                    comm.Recv(buf, source=rank, tag=rank)
                    ukr[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]=buf 

    ###        
    ###
    if rem_z==0:
        uk_loc2_r=np.empty([nx,ny,zsz], dtype=uk_loc_r.dtype)
        if my_rank==0:
            uk_loc2_r=ukr[:,:,my_rank*(zsz):(my_rank+1)*(zsz)]
            for rank in np.arange(1,p):
                buf=np.ascontiguousarray(np.copy(ukr[:,:,(rank*zsz):((rank+1)*zsz)]))
                comm.Send(buf, dest=rank, tag=rank*2)
        else:
            comm.Recv(uk_loc2_r, source=0, tag=my_rank*2)
    else:
        if my_rank in np.arange(rem_z):
            uk_loc2_r=np.empty([nx,ny,zsz+1], dtype=uk_loc_r.dtype)
        else:
            uk_loc2_r=np.empty([nx,ny,zsz], dtype=uk_loc_r.dtype)
        if my_rank==0:
            uk_loc2_r=ukr[:,:,my_rank*(zsz+1):(my_rank+1)*(zsz+1)]
            if rem_z==1:
                for rank in np.arange(rem_z,p):
                    buf=np.ascontiguousarray(np.copy(ukr[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]))
                    comm.Send(buf, dest=rank, tag=rank*2)
            else:
                for rank in np.arange(1,rem_z):
                    buf=np.ascontiguousarray(np.copy(ukr[:,:,rank*(zsz+1):(rank+1)*(zsz+1)]))
                    comm.Send(buf, dest=rank, tag=rank*2)
                
                for rank in np.arange(rem_z,p):
                    buf=np.ascontiguousarray(np.copy(ukr[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]))
                    comm.Send(buf, dest=rank, tag=rank*2)
        else:
            comm.Recv(uk_loc2_r, source=0, tag=my_rank*2)
    del ukr
    #########################################################
            
    if rem_y==0:
        if my_rank!=0:
            uki=None
            comm.Send(uk_loc_i, dest=0, tag=my_rank*3)
        else:
            uki=np.empty([nx,ny,nz], dtype=uk_loc_i.dtype)
            uki[:,(my_rank*(ysz)):((my_rank+1)*(ysz)),:]=uk_loc_i
            for rank in np.arange(1,p):
                buf=np.empty([nx,ysz,nz], dtype=uk_loc_i.dtype)
                comm.Recv(buf, source=rank, tag=rank*3)
                uki[:,(rank*ysz):((rank+1)*ysz),:]=buf 
    else:
        if my_rank!=0:
            uki=None
            comm.Send(uk_loc_i, dest=0, tag=my_rank*3)
        else:
            uki=np.empty([nx,ny,nz], dtype=uk_loc_i.dtype)
            uki[:,(my_rank*(ysz+1)):((my_rank+1)*(ysz+1)),:]=uk_loc_i
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.empty([nx,ysz,nz], dtype=uk_loc_i.dtype)
                    comm.Recv(buf, source=rank, tag=rank*3)
                    uki[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]=buf 
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.empty([nx,ysz+1,nz], dtype=uk_loc_i.dtype)
                    comm.Recv(buf, source=rank, tag=rank*3)
                    uki[:,(rank*(ysz+1)):((rank+1)*(ysz+1)),:]=buf
                
                for rank in np.arange(rem_y,p):
                    buf=np.empty([nx,ysz,nz], dtype=uk_loc_i.dtype)
                    comm.Recv(buf, source=rank, tag=rank*3)
                    uki[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]=buf 

    ###        
    ###
    if rem_z==0:
        uk_loc2_i=np.empty([nx,ny,zsz], dtype=uk_loc_i.dtype)
        if my_rank==0:
            uk_loc2_i=uki[:,:,my_rank*(zsz):(my_rank+1)*(zsz)]
            for rank in np.arange(1,p):
                buf=np.ascontiguousarray(np.copy(uki[:,:,(rank*zsz):((rank+1)*zsz)]))
                comm.Send(buf, dest=rank, tag=rank*4)
        else:
            comm.Recv(uk_loc2_i, source=0, tag=my_rank*4)
    else:
        if my_rank in np.arange(rem_z):
            uk_loc2_i=np.empty([nx,ny,zsz+1], dtype=uk_loc_i.dtype)
        else:
            uk_loc2_i=np.empty([nx,ny,zsz], dtype=uk_loc_i.dtype)
        if my_rank==0:
            uk_loc2_i=uki[:,:,my_rank*(zsz+1):(my_rank+1)*(zsz+1)]
            if rem_z==1:
                for rank in np.arange(rem_z,p):
                    buf=np.ascontiguousarray(np.copy(uki[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]))
                    comm.Send(buf, dest=rank, tag=rank*4)
            else:
                for rank in np.arange(1,rem_z):
                    buf=np.ascontiguousarray(np.copy(uki[:,:,rank*(zsz+1):(rank+1)*(zsz+1)]))
                    comm.Send(buf, dest=rank, tag=rank*4)
                
                for rank in np.arange(rem_z,p):
                    buf=np.ascontiguousarray(np.copy(uki[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]))
                    comm.Send(buf, dest=rank, tag=rank*4)
        else:
            comm.Recv(uk_loc2_i, source=0, tag=my_rank*4)
    del uki
    #########################################################
    uk_loc2=uk_loc2_r+1j*uk_loc2_i
    del uk_loc_i, uk_loc_r, uk_loc, uk_loc2_r, uk_loc2_i

    
    N=int(4*np.floor(Re_tau))
    y2=np.linspace(0,2,N)
    
    if rem_z==0:
        kz_loc=kz[my_rank*(zsz):(my_rank+1)*(zsz)]
    else:
        if my_rank in np.arange(rem_z):
            kz_loc=kz[my_rank*(zsz+1):(my_rank+1)*(zsz+1)]
        else:
            kz_loc=kz[my_rank*zsz+rem_z:(my_rank+1)*zsz+rem_z]

    ky=scipy.fft.fftfreq((2*N-2), (2*Ly)/(2*N-2))*2*np.pi

    [a,b,c]=np.shape(uk_loc2)    
    for i in np.arange(nx):
        for k in np.arange(c):
            temp=uk_loc2[i,:,k]
            temp_i=np.interp(y2,y,temp)
            temp_r=np.empty((2*N-2), dtype=uk_loc2.dtype)
            temp_r[(N-2):(2*N-2)]=temp_i
            for n in np.arange((N-2)):
                temp_r[n]=-temp_i[((N-3)-n+1)]
            temp_r2=scipy.fft.fft(temp_r, n=(2*N-2))
            kappa=np.sqrt(kx[i]**2+ky**2+kz_loc[k]**2)*L/2
            Tb=(2*(kappa**2))*(np.exp(-(kappa**2)))
            temp_r3=np.sqrt(2/L)*Tb*temp_r2
            temp_r=scipy.fft.ifft(temp_r3, n=(2*N-2))
            temp_i=temp_r[(N-2):(2*N-2)]
            temp=np.interp(y,y2,temp_i)
            uk_loc2[i,:,k]=temp.astype(np.csingle)
            del temp, temp_i, temp_r, temp_r2, temp_r3   
     
  
    comm.Barrier()    
    uk_loc2_r=np.ascontiguousarray(np.copy(np.real(uk_loc2)))
    uk_loc2_i=np.ascontiguousarray(np.copy(np.imag(uk_loc2)))

    #########################################################
    if rem_z==0:
        if my_rank!=0:
            ukr=None
            comm.Send(uk_loc2_r, dest=0, tag=my_rank*5)
        else:
            ukr=np.empty([nx,ny,nz], dtype=uk_loc2_r.dtype)
            ukr[:,:,(my_rank*(zsz)):((my_rank+1)*(zsz))]=uk_loc2_r
            for rank in np.arange(1,p):
                buf=np.empty([nx,ny,zsz], dtype=uk_loc2_r.dtype)
                comm.Recv(buf, source=rank, tag=rank*5)
                ukr[:,:,(rank*zsz):((rank+1)*zsz)]=buf
    else:    
        if my_rank!=0:
            ukr=None
            comm.Send(uk_loc2_r, dest=0, tag=my_rank*5)
        else:
            ukr=np.empty([nx,ny,nz], dtype=uk_loc2_r.dtype)
            ukr[:,:,(my_rank*(zsz+1)):((my_rank+1)*(zsz+1))]=uk_loc2_r
            if rem_z==1:
                for rank in np.arange(rem_z,p):
                    buf=np.empty([nx,ny,zsz], dtype=uk_loc2_r.dtype)
                    comm.Recv(buf, source=rank, tag=rank*5)
                    ukr[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]=buf 
            else:
                for rank in np.arange(1,rem_z):
                    buf=np.empty([nx,ny,zsz+1], dtype=uk_loc2_r.dtype)
                    comm.Recv(buf, source=rank, tag=rank*5)
                    ukr[:,:,(rank*(zsz+1)):((rank+1)*(zsz+1))]=buf
                
                for rank in np.arange(rem_z,p):
                    buf=np.empty([nx,ny,zsz], dtype=uk_loc2_r.dtype)
                    comm.Recv(buf, source=rank, tag=rank*5)
                    ukr[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]=buf  
    

    ###
    if rem_y==0:
        uk_loc4_r=np.empty([nx,ysz,nz], dtype=uk_loc2_r.dtype)
        if my_rank==0:
            uk_loc4_r=ukr[:,my_rank*(ysz):(my_rank+1)*(ysz),:]
            for rank in np.arange(1,p):
                buf=np.ascontiguousarray(np.copy(ukr[:,(rank*ysz):((rank+1)*ysz),:]))
                comm.Send(buf, dest=rank, tag=rank*6)
        else:
            comm.Recv(uk_loc4_r, source=0, tag=my_rank*6)
    else:
        if  my_rank in np.arange(rem_y):
            uk_loc4_r=np.empty([nx,ysz+1,nz], dtype=uk_loc2_r.dtype)
        else:
            uk_loc4_r=np.empty([nx,ysz,nz], dtype=uk_loc2_r.dtype)
        if my_rank==0:
            uk_loc4_r=ukr[:,my_rank*(ysz+1):(my_rank+1)*(ysz+1),:]
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(ukr[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank*6)
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.ascontiguousarray(np.copy(ukr[:,rank*(ysz+1):(rank+1)*(ysz+1),:]))
                    comm.Send(buf, dest=rank, tag=rank*6)
                
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(ukr[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank*6)
        else:
            comm.Recv(uk_loc4_r, source=0, tag=my_rank*6)
    del ukr
    #########################################################
            
    if rem_z==0:
        if my_rank!=0:
            uki=None
            comm.Send(uk_loc2_i, dest=0, tag=my_rank*7)
        else:
            uki=np.empty([nx,ny,nz], dtype=uk_loc2_i.dtype)
            uki[:,:,(my_rank*(zsz)):((my_rank+1)*(zsz))]=uk_loc2_i
            for rank in np.arange(1,p):
                buf=np.empty([nx,ny,zsz], dtype=uk_loc2_i.dtype)
                comm.Recv(buf, source=rank, tag=rank*7)
                uki[:,:,(rank*zsz):((rank+1)*zsz)]=buf
    else:    
        if my_rank!=0:
            uki=None
            comm.Send(uk_loc2_i, dest=0, tag=my_rank*7)
        else:
            uki=np.empty([nx,ny,nz], dtype=uk_loc2_i.dtype)
            uki[:,:,(my_rank*(zsz+1)):((my_rank+1)*(zsz+1))]=uk_loc2_i
            if rem_z==1:
                for rank in np.arange(rem_z,p):
                    buf=np.empty([nx,ny,zsz], dtype=uk_loc2_i.dtype)
                    comm.Recv(buf, source=rank, tag=rank*7)
                    uki[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]=buf 
            else:
                for rank in np.arange(1,rem_z):
                    buf=np.empty([nx,ny,zsz+1], dtype=uk_loc2_i.dtype)
                    comm.Recv(buf, source=rank, tag=rank*7)
                    uki[:,:,(rank*(zsz+1)):((rank+1)*(zsz+1))]=buf
                
                for rank in np.arange(rem_z,p):
                    buf=np.empty([nx,ny,zsz], dtype=uk_loc2_i.dtype)
                    comm.Recv(buf, source=rank, tag=rank*7)
                    uki[:,:,(rank*zsz+rem_z):((rank+1)*zsz+rem_z)]=buf  
    

    ###
    if rem_y==0:
        uk_loc4_i=np.empty([nx,ysz,nz], dtype=uk_loc2_i.dtype)
        if my_rank==0:
            uk_loc4_i=uki[:,my_rank*(ysz):(my_rank+1)*(ysz),:]
            for rank in np.arange(1,p):
                buf=np.ascontiguousarray(np.copy(uki[:,(rank*ysz):((rank+1)*ysz),:]))
                comm.Send(buf, dest=rank, tag=rank*8)
        else:
            comm.Recv(uk_loc4_i, source=0, tag=my_rank*8)
    else:
        if  my_rank in np.arange(rem_y):
            uk_loc4_i=np.empty([nx,ysz+1,nz], dtype=uk_loc2_i.dtype)
        else:
            uk_loc4_i=np.empty([nx,ysz,nz], dtype=uk_loc2_i.dtype)
        if my_rank==0:
            uk_loc4_i=uki[:,my_rank*(ysz+1):(my_rank+1)*(ysz+1),:]
            if rem_y==1:
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(uki[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank*8)
            else:
                for rank in np.arange(1,rem_y):
                    buf=np.ascontiguousarray(np.copy(uki[:,rank*(ysz+1):(rank+1)*(ysz+1),:]))
                    comm.Send(buf, dest=rank, tag=rank*8)
                
                for rank in np.arange(rem_y,p):
                    buf=np.ascontiguousarray(np.copy(uki[:,(rank*ysz+rem_y):((rank+1)*ysz+rem_y),:]))
                    comm.Send(buf, dest=rank, tag=rank*8)
        else:
            comm.Recv(uk_loc4_i, source=0, tag=my_rank*8)
    #########################################################
    uk_loc4=uk_loc4_r+1j*uk_loc4_i
    del uki, uk_loc4_i, uk_loc4_r

    out=scipy.fft.ifft2(uk_loc4, s=[nx, nz], axes=(0,2))  
    del uk_loc4
    ub0=np.ascontiguousarray(np.copy(np.real(out)))
    ub=ub0.astype(np.float32)
    #if my_rank==0:
    #    print(time.time())
    return ub
#--------------------------------------------------------------------------------------------------------------------------------------------#
def fluc(input):
    (a,b,c)=np.shape(input)
    fluct=np.empty_like(input)
    for i in np.arange(b):
        mean=np.mean(input[:,i,:])
        fluct[:,i,:]=np.squeeze(input[:,i,:])-mean*np.ones([a,c])
    return fluct

        







