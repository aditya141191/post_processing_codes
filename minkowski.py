import numpy as np
import os
import scipy
import time

def swap(idir, BR, nx, ny, nz):
    """
    Swaps the elements of a 3D array along different dimensions.

    Args:
        idir: An integer indicating the type of swap operation (1-4).
        BR: The input 3D NumPy array.
        nx: Number of elements in the x-dimension.
        ny: Number of elements in the y-dimension.
        nz: Number of elements in the z-dimension.

    Returns:
        A tuple containing the modified array BR and the updated idir.
    """

    if idir == 1:  # x-swap
        BR = np.flip(BR, axis=0)
        idir = 2
    elif idir == 2:  # x and y swap
        BR = np.flip(BR, axis=1)
        idir = 3
    elif idir == 3:  # x, y and z swap
        BR = np.flip(BR, axis=2)
        idir = 4
    elif idir == 4:  # all swap
        BR = np.flip(BR, axis=(0,1,2))
        idir = 5
    else:
      print("Error: Invalid idir value.")
      return BR, idir

    return BR, idir

def interp3_np(x1, y1, z1, CR, x2, y2, z2):
    """
    Performs 3D interpolation using np.interp along each axis.

    Args:
        x1, y1, z1: 1D arrays defining the coordinates of the source grid.
        CR: 3D array of data on the source grid.
        x2, y2, z2: 1D arrays defining the coordinates of the target grid.

    Returns:
        3D array of interpolated data on the target grid.
    """
    # Interpolation along Z-axis (depth)
    temp_z = np.zeros((len(x1), len(y1), len(z2)))
    for i in range(len(x1)):
        for j in range(len(y1)):
            temp_z[i, j, :] = np.interp(z2, z1, CR[i, j, :])  # 1D interpolation along Z
    
    # Interpolation along Y-axis
    temp_y = np.zeros((len(x1), len(y2), len(z2)))
    for i in range(len(x1)):
        for k in range(len(z2)):
            temp_y[i, :, k] = np.interp(y2, y1, temp_z[i, :, k])  # 1D interpolation along Y
    
    # Interpolation along X-axis
    CR2 = np.zeros((len(x2), len(y2), len(z2)))
    for j in range(len(y2)):
        for k in range(len(z2)):
            CR2[:, j, k] = np.interp(x2, x1, temp_y[:, j, k])  # 1D interpolation along X

    return CR2

def march(BR, cnt, fiperi, fjperi, fkperi, nx, ny, nz, flag1, thres):
    """
    3D region growing algorithm.

    Args:
        BR: 3D NumPy array.
        cnt: Counter variable (integer).
        fiperi, fjperi, fkperi: Boundary flags (integers, initially 0).
        nx, ny, nz: Dimensions of BR.
        flag1: Value marking the region to grow.
        thres: Value of elements to be included in the region.

    Returns:
        Tuple: Updated BR, cnt, fiperi, fjperi, fkperi.
    """

    #First loop: j=1 (first plane)
    for kk in range(nz):
        for ii in range(nx):
            if BR[ii, 0, kk] == flag1:  #Note the change to 0 instead of jj=1
                fjperi = 1
                for kkk in [-1, 0, 1]:
                    for iii in [-1, 0, 1]:
                        for jjj in [0,1]: #only positive j direction
                            iiii = (ii + iii)%nx
                            jjjj = (0 + jjj)%ny
                            kkkk = (kk + kkk)%nz
                            if BR[iiii,jjjj,kkkk] == thres:
                                cnt += 1
                                BR[iiii,jjjj,kkkk] = flag1
                                if kk == 0 or kk == nz -1:
                                    fkperi = 1
                                if ii == 0 or ii == nx -1:
                                    fiperi = 1

    #Second loop: 1 < j < ny-1 (inner planes)
    for kk in range(nz):
        for jj in range(1, ny - 1):
            for ii in range(nx):
                if BR[ii, jj, kk] == flag1:
                    for kkk in [-1, 0, 1]:
                        for jjj in [-1, 0, 1]:
                            for iii in [-1, 0, 1]:
                                iiii = (ii + iii)%nx
                                jjjj = (jj + jjj)%ny
                                kkkk = (kk + kkk)%nz
                                if BR[iiii,jjjj,kkkk] == thres:
                                    cnt += 1
                                    BR[iiii, jjjj, kkkk] = flag1
                                    if kk == 0 or kk == nz - 1:
                                        fkperi = 1
                                    if ii == 0 or ii == nx - 1:
                                        fiperi = 1

    #Third loop: j=ny (last plane)
    for kk in range(nz):
        for ii in range(nx):
            if BR[ii, ny - 1, kk] == flag1: #Note the change to ny-1 instead of jj=ny
                fjperi = 1
                for kkk in [-1, 0, 1]:
                    for iii in [-1, 0, 1]:
                        for jjj in [-1,0]: #only negative j direction
                            iiii = (ii + iii)%nx
                            jjjj = (ny -1 + jjj)%ny #Note the change to ny-1 instead of jj=ny
                            kkkk = (kk + kkk)%nz
                            if BR[iiii,jjjj,kkkk] == thres:
                                cnt += 1
                                BR[iiii, jjjj, kkkk] = flag1
                                if kk == 0 or kk == nz - 1:
                                    fkperi = 1
                                if ii == 0 or ii == nx - 1:
                                    fiperi = 1

    return BR, cnt, fiperi, fjperi, fkperi

def march_idx(BR, axis, size, thres, flag1):
    if axis==0:
        if np.any(BR[0,:,:] >= thres):
            idx = 1  
            BR[0,:,:][BR[0,:,:] >= thres] = flag1
            flag_update=True
        else:
            return None

        i=0
        while flag_update==True and i<size:
            mask = BR[i,:,:]==flag1
            neighbour = [-1, 0, 1]
            for neighbour in [-1, 0, 1]:
                if np.any(BR[i+1,:,:][mask+neighbour])>=thres:
                    mask2=BR[i+1,:,:][mask+neighbour]>=thres
                    BR[mask2]=flag1
                    idx=i+1
                    i+=1
                else:
                    flag_update=False

    if axis==1:
        if np.any(BR[:,0,:] >= thres):
            idx = 1
            BR[:,0,:][BR[:,0,:] >= thres] = flag1
            flag_update=True
        else:
            return None

        i=0
        while flag_update==True and i<size:
            mask = BR[:,i,:]==flag1
            neighbour = [-1, 0, 1]
            for neighbour in [-1, 0, 1]:
                if np.any(BR[:,i+1,:][mask+neighbour])>=thres:
                    mask2=BR[:,i+1,:][mask+neighbour]>=thres
                    BR[mask2]=flag1
                    idx=i+1
                    i+=1
                else:
                    flag_update=False
    
    if axis==2:
        if np.any(BR[:,:,0] >= thres):
            idx=1
            BR[:,:,0][BR[:,:,0] >= thres] = flag1
            flag_update=True
        else:
            return None
                
        i=0
        while flag_update==True and i<size:
            mask = BR[:,:,i]==flag1
            neighbour = [-1, 0, 1]
            for neighbour in [-1, 0, 1]:
                if np.any(BR[:,:,i+1][mask+neighbour])>=thres:
                    mask2=BR[:,:,i+1][mask+neighbour]>=thres
                    BR[mask2]=flag1
                    idx=i+1
                    i+=1
                else:
                    flag_update=False
    
    return idx
    
def pick2_new(CR, nx, ny, nz, N, thres, flag1, flag2, fiperi, fjperi, fkperi, LCA, FAC, lmin):
    ND = (nx - N) // 2
    key = 0
    AS = np.copy(CR)
    DI = DJ = DK = Dmin = Dmid = Dmax = imin = imax = jmin = jmax = kmin = kmax = 0

    BR = np.copy(CR)




    def _find_extremes_single_axis(BR, axis, size, thres, peri_flag):
        """Helper function to find min/max indices along a single axis."""
        if peri_flag == 0:  # Non-periodic
            min_idx = np.argmax(np.any(BR >= thres, axis=tuple(i for i in range(3) if i != axis))) + 1 if np.any(BR >= thres) else None
            max_idx = size - np.argmax(np.any(np.flip(BR, axis=axis) >= thres, axis=tuple(i for i in range(3) if i != axis))) if np.any(np.flip(BR, axis=axis) >= thres) else None

        else:  # Periodic
            # max_idx=march_idx(BR, axis, size, thres, flag1)
            # min_idx=march_idx(np.flip(BR, axis), axis, size, thres, flag1)
            # min_idx=-min_idx if min_idx is not None else None
            max_idx = np.argmax(np.all(BR < thres, axis=tuple(i for i in range(3) if i != axis))) if np.any(BR < thres) else None
            min_idx = - np.argmax(np.all(np.flip(BR, axis=axis) < thres, axis=tuple(i for i in range(3) if i != axis))) if np.any(np.flip(BR, axis=axis) < thres)+1 else None


        return min_idx, max_idx
    
   
    def handle_boundary_conditions(imin, imax, jmin, jmax, kmin, kmax, nx, ny, nz, fiperi, fjperi, fkperi):
        """Handles boundary conditions and returns updated flags and dimensions."""

        if any(idx == None for idx in [imin, imax, jmin, jmax, kmin, kmax]):
            return 6, None, None, None

        boundary_flags = [(fiperi, imin, imax, nx), (fjperi, jmin, jmax, ny), (fkperi, kmin, kmax, nz)]
        for flag, min_idx, max_idx, size in boundary_flags:
            if flag == 1 and min_idx == 1 and max_idx == size:
                flag = 2
            elif flag == 1 and (min_idx == size or max_idx == 1):
                flag = 5


        return fiperi, None, None, None
    

    imin, imax = _find_extremes_single_axis(BR, 0, nx, thres, fiperi)
    jmin, jmax = _find_extremes_single_axis(BR, 1, ny, thres, fjperi)
    kmin, kmax = _find_extremes_single_axis(BR, 2, nz, thres, fkperi)  
    print(imin, imax, jmin, jmax, kmin, kmax, fiperi, fjperi, fkperi)
    fiperi,_,_,_=handle_boundary_conditions(imin, imax, jmin, jmax, kmin, kmax, nx, ny, nz, fiperi, fjperi, fkperi)

    if fiperi in (2,5,6):
        print("fiper is in (2,5,6)",fiperi)
        return AS, Dmin, Dmid, Dmax, fiperi, fjperi, fkperi, DI, DJ, DK

    DI = (imax - imin + 1) #+ (nx if fiperi == 1 else 0)
    DJ = (jmax - jmin + 1) #+ (ny if fjperi == 1 else 0)
    DK = (kmax - kmin + 1) #+ (nz if fkperi == 1 else 0)

    if any(dim > size for dim, size in zip([DI, DJ, DK], [nx, ny, nz])):
        fiperi = 3
        return AS, Dmin, Dmid, Dmax, fiperi, fjperi, fkperi, DI, DJ, DK

    Dmin, Dmid, Dmax = sorted([DI, DJ, DK])
    if np.any(np.array([Dmin, Dmid, Dmax]) <= lmin):
        fiperi = 4
        return AS, Dmin, Dmid, Dmax, fiperi, fjperi, fkperi, DI, DJ, DK


    # if fiperi == 1:
    #     imin = -(nx - imin)
    # if fjperi == 1:
    #     jmin = -(ny - jmin)
    # if fkperi == 1:
    #     kmin = -(nz - kmin)

    ishift = (nx // 2) - ((imax + imin) // 2)
    jshift = (ny // 2) - ((jmax + jmin) // 2)
    kshift = (nz // 2) - ((kmax + kmin) // 2)

    ii = np.arange(nx) - ishift
    jj = np.arange(ny) - jshift
    kk = np.arange(nz) - kshift
    ii = np.mod(ii, nx)
    jj = np.mod(jj, ny)
    kk = np.mod(kk, nz)
    AS = CR[ii[:,None,None], jj[None,:,None], kk[None,None,:]]

    return AS, Dmin, Dmid, Dmax, fiperi, fjperi, fkperi, DI, DJ, DK

def minsub(AR, DI, DJ, DK, x, y, z, a, thres):
    nx, ny, nz = AR.shape

    imin = max(1, int(np.floor(nx / 2) - np.floor(DI / 2) - 1))
    imax = min(nx, int(np.floor(nx / 2) + np.floor(DI / 2) + 3))
    jmin = max(1, int(np.floor(ny / 2) - np.floor(DJ / 2) - 1))
    jmax = min(ny, int(np.floor(ny / 2) + np.floor(DJ / 2) + 3))
    kmin = max(1, int(np.floor(nz / 2) - np.floor(DK / 2) - 1))
    kmax = min(nz, int(np.floor(nz / 2) + np.floor(DK / 2) + 3))

    # print([imin, imax, jmin, jmax, kmin, kmax])
    CR = AR[imin - 1:imax, jmin - 1:jmax, kmin - 1:kmax]  # Adjust indexing for Python
    x1 = x[imin - 1:imax]
    y1 = y[jmin - 1:jmax]
    z1 = z[kmin - 1:kmax]

    Lx = x1[-1] - x1[0]
    Ly = y1[-1] - y1[0]
    Lz = z1[-1] - z1[0]

    Lmax = max(Lx, Ly, Lz)

    Nx = int(np.floor(Lx / a) + 1)
    x2 = np.linspace(x1[0], x1[0] + (Nx - 1) * a, Nx)
    Ny = int(np.floor(Ly / a) + 1)
    y2 = np.linspace(y1[0], y1[0] + (Ny - 1) * a, Ny)
    Nz = int(np.floor(Lz / a) + 1)
    z2 = np.linspace(z1[0], z1[0] + (Nz - 1) * a, Nz)

    CR2=interp3_np(x1, y1, z1, CR, x2, y2, z2)

    BR = CR2
    N0 = N1 = N2 = N3 = 0
    EPS = 1e-5
    A = np.zeros(7)

    # print("CP6")
    N3 = np.sum(BR > thres)
    # print("N3",N3)
    # for k in range(Nz):
    #     for j in range(Ny):
    #         for i in range(Nx):
    #             if BR[i, j, k] > thres:
    #                 N3 += 1

    # print("CP7")
    BR = np.copy(CR2)
    t=time.time()
    for k in range(1, Nz - 1):
        if k%10==0:
            print(f"Time:{time.time()-t}, Loop copmletion:{100*k/Nz}%") 
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if BR[i, j, k] > thres:
                    A[6] = BR[i, j, k]
                    A[0] = BR[i - 1, j, k]
                    A[1] = BR[i + 1, j, k]
                    A[2] = BR[i, j - 1, k]
                    A[3] = BR[i, j + 1, k]
                    A[4] = BR[i, j, k - 1]
                    A[5] = BR[i, j, k + 1]
                    for ii in range(6):
                        if thres > A[ii]:
                            N2 += 1
                    BR[i, j, k] = 0

    # print("N2",N2)
    # print("CP8")
    BR = np.copy(CR2)
    B = np.zeros((3, 3, 3))
    A = np.zeros(7)
    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if BR[i, j, k] > thres:
                    A[6] = BR[i, j, k]
                    for ii in [-1, 0, 1]:
                        for jj in [-1, 0, 1]:
                            for kk in [-1, 0, 1]:
                                B[ii + 1, jj + 1, kk + 1] = BR[i + ii, j + jj, k + kk]
                    for jj in [-1, 1]:
                        for ii in [-1, 1]:
                            A[0] = B[ii + 1, 1, 1]
                            A[1] = B[ii + 1, jj + 1, 1]
                            A[2] = B[1, jj + 1, 1]
                            if thres > A[0] and thres > A[1] and thres > A[2]:
                                N1 += 1
                    for kk in [-1, 1]:
                        for ii in [-1, 1]:
                            A[0] = B[1, 1, kk + 1]
                            A[1] = B[ii + 1, 1, kk + 1]
                            A[2] = B[ii + 1, 1, 1]
                            if thres > A[0] and thres > A[1] and thres > A[2]:
                                N1 += 1
                    for kk in [-1, 1]:
                        for jj in [-1, 1]:
                            A[0] = B[1, jj + 1, 1]
                            A[1] = B[1, jj + 1, kk + 1]
                            A[2] = B[1, 1, kk + 1]
                            if thres > A[0] and thres > A[1] and thres > A[2]:
                                N1 += 1
                    BR[i, j, k] = 0
    # print("N1",N1)
    # print("CP9")
    BR = np.copy(CR2)
    A = np.zeros(7)
    B = np.zeros((3, 3, 3))
    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if BR[i, j, k] > thres:
                    A[6] = BR[i, j, k]
                    for ii in [-1, 0, 1]:
                        for jj in [-1, 0, 1]:
                            for kk in [-1, 0, 1]:
                                B[ii + 1, jj + 1, kk + 1] = BR[i + ii, j + jj, k + kk]
                    for kk in [-1, 1]:
                        for jj in [-1, 1]:
                            for ii in [-1, 1]:
                                A[0] = B[ii + 1, jj + 1, 1]
                                A[1] = B[ii + 1, 1, 1]
                                A[2] = B[ii + 1, jj + 1, kk + 1]
                                A[3] = B[ii + 1, 1, kk + 1]
                                A[4] = B[1, jj + 1, kk + 1]
                                A[5] = B[1, 1, kk + 1]
                                A[6] = B[1, jj + 1, 1]
                                if thres > A[0] and thres > A[1] and thres > A[2] and thres > A[3] and thres > A[4] and thres > A[5] and thres > A[6]:
                                    N0 += 1
                    BR[i, j, k] = 0
    print("N0",N0)
    N = 1 / a
    N0 = N0 / N ** 3
    N1 = N1 / N ** 3
    N2 = N2 / N ** 3
    N3 = N3 / N ** 3

    V0 = N3
    V1 = (2 / (9 * a)) * (N2 - 3 * N3)
    V2 = (2 / (9 * a ** 2)) * (N1 - 2 * N2 + 3 * N3)
    V3 = (a ** -3) * (N0 - N1 + N2 - N3)
    V3 = 2 * V3
    print("V0,V1,V2,V3",V0,V1,V2,V3)

    MT = V0 / (2 * V1) if V1 !=0 else 0 #Handle division by zero
    MW = 2 * V1 / (np.pi * V2) if V2 != 0 else 0 #Handle division by zero
    ML = (3 * V2) / (4 * (2 - V3 / 2)) if abs(V3 / V2) < EPS or (V3 / V2) < 0 else (3 * V2) / (2 * V3) if V3 != 0 else 0 #Handle division by zero and conditional logic
    MP = (MW - MT) / (MW + MT) if (MW+MT)!= 0 else 0 #Handle division by zero
    MF = (ML - MW) / (ML + MW) if (ML+MW)!= 0 else 0 #Handle division by zero

    return MT, MW, ML, MP, MF, V0, V1, V2, V3

def count_elements(val, thres):
    Nx, Ny, Nz = val.shape
    
    # 0D: Vertices
    N0 = np.sum(val > thres)

    # 1D: Edges
    N1x = np.sum((val[:-1, :, :] > thres) & (val[1:, :, :] > thres))
    N1y = np.sum((val[:, :-1, :] > thres) & (val[:, 1:, :] > thres))
    N1z = np.sum((val[:, :, :-1] > thres) & (val[:, :, 1:] > thres))
    N1 = N1x + N1y + N1z

    # 2D: Faces
    N2xy = np.sum((val[:-1, :-1, :] > thres) & (val[1:, :-1, :] > thres) & 
                  (val[:-1, 1:, :] > thres) & (val[1:, 1:, :] > thres))
    N2yz = np.sum((val[:, :-1, :-1] > thres) & (val[:, 1:, :-1] > thres) &
                  (val[:, :-1, 1:] > thres) & (val[:, 1:, 1:] > thres))
    N2zx = np.sum((val[:-1, :, :-1] > thres) & (val[1:, :, :-1] > thres) &
                  (val[:-1, :, 1:] > thres) & (val[1:, :, 1:] > thres))
    N2 = N2xy + N2yz + N2zx

    # 3D: Volumes
    N3 = np.sum((val[:-1, :-1, :-1] > thres) & (val[1:, :-1, :-1] > thres) &
                (val[:-1, 1:, :-1] > thres) & (val[1:, 1:, :-1] > thres) &
                (val[:-1, :-1, 1:] > thres) & (val[1:, :-1, 1:] > thres) &
                (val[:-1, 1:, 1:] > thres) & (val[1:, 1:, 1:] > thres))

    return N0, N1, N2, N3

def pick(var, nx, ny, nz, thres, strmax, fname, fpath, lmin, zone, a, x, y, z, sub_rank):
    tm=time.time()
    flag1 = 100000
    flag2 = 200000
    # BR = np.copy(var)
    N = nx
    EPS_V3 = 0.01
    output = np.zeros((strmax, 13))

    if not os.path.exists(os.path.join(fpath, fname)):
        print("Making directory",os.path.join(fpath, fname) )
        os.makedirs(os.path.join(fpath, fname))
        

    BR = np.where(var > thres, thres, 0)

    if zone == 1:
        BR[:,0,:]=0

    # print("CP0")
    totcnt = 0
    strcnt = 0
    for k in range(nz):
        # print(k)
        for j in range(ny):
            for i in range(nx):
                fiperi = fjperi = fkperi = 0
                if BR[i, j, k] == thres:
                    strcnt += 1
                    totcnt += 1
                    BR[i, j, k] = flag1
                    cnt = 1
                    cnto = cnt
                    idir = 1
                    while idir < 5:
                        # print("CP1")
                        BR, cnt, fiperi, fjperi, fkperi = march(BR, cnt, fiperi, fjperi, fkperi, nx, ny, nz, flag1, thres)
                        if cnt > cnto:
                            cnto = cnt
                        else:
                            # print("CP2")
                            BR, idir = swap(idir, BR, nx, ny, nz)

                    CR=np.where(np.logical_or(var < thres, BR == flag1), var, 0)
                    # print(np.any(BR==thres))
                    BR[BR==flag1]=flag2
                    # print(np.any(BR==thres))
                    # CR = np.zeros_like(BR)
                    # for kk in range(nz):
                    #     for jj in range(ny):
                    #         for ii in range(nx):
                    #             if BR[ii, jj, kk] == flag1:
                    #                 CR[ii, jj, kk] = var[ii, jj, kk]
                    #                 BR[ii, jj, kk] = flag2
                    #             else:
                    #                 CR[ii, jj, kk] = 0 if var[ii, jj, kk] >= thres else var[ii, jj, kk]

                    LCA = 1 / nx
                    FAC = 1
                    if fjperi == 1 or fiperi == 1 or fkperi == 1:
                        print(f"Boundary structure discarded")
                        strcnt -= 1
                    else:
                        # print("CP3")
                        AS, dmin, dmid, dmax, fiperi, fjperi, fkperi, DI, DJ, DK = pick2_new(CR, nx, ny, nz, N, thres, flag1, flag2, fiperi, fjperi, fkperi, LCA, FAC, lmin)
                        ASmax = np.max(AS)

                        if fiperi in [2,3,4,5,6] or fjperi in [2,5] or fkperi in [2,5] or ASmax < thres:
                            print("Structure discarded",(fiperi, fjperi, fkperi,i,j,k))
                            strcnt -= 1
                        else:
                            print("Running minsub",i,j,k)
                            print(f"Time taken to run pick:{tm-time.time()}")
                            MT, MW, ML, MP, MF, V0, V1, V2, V3 = minsub(AS, DI, DJ, DK, x, y, z, a, thres)
                            output[strcnt - 1, 0] = totcnt
                            output[strcnt - 1, 1] = V0
                            output[strcnt - 1, 2] = V1
                            output[strcnt - 1, 3] = V2
                            output[strcnt - 1, 4] = V3
                            output[strcnt - 1, 5] = MT
                            output[strcnt - 1, 6] = MW
                            output[strcnt - 1, 7] = ML
                            output[strcnt - 1, 8] = MP
                            output[strcnt - 1, 9] = MF
                            output[strcnt - 1, 10] = dmin
                            output[strcnt - 1, 11] = dmid
                            output[strcnt - 1, 12] = dmax

                            # fname2 = os.path.join(fpath, fname, 'str' + "{:03d}".format(strcnt))
                            # np.savez_compressed(fname2, CR=CR, AS=AS)

                            fname2 = os.path.join(fpath, fname, 'str' + "{:03d}".format(strcnt))
                            print("CP5")
                            # Save CR and AS to separate binary files
                            scipy.io.savemat(fname2 + ".mat", {'CR': CR, 'AS': AS})
                            # CR.tofile(fname2 + "_CR.bin")
                            # AS.tofile(fname2 + "_AS.bin")

                            np.save(fname2+"CR.npy", CR)
                            np.save(fname2+"AS.npy", AS)

                            fname3 = os.path.join(fpath, 'rank' + "{:03d}".format(sub_rank))
                            with open(fname3+".txt", "a") as f:
                                np.savetxt(f, np.reshape(output[strcnt-1,:],(1,13)), fmt="%.4f")

                            if strcnt == strmax:
                                return output
                        

    if strcnt==0:
        print("No structures detected")
    
    return output
