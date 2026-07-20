raw_data_path = "/scratch/aseaditya/Jimenez_data/Re4000/results/scaling/"
output_path = "/scratch/aseaditya/Jimenez_data/Re4000/minkowski/"

with open('/scratch/aseaditya/Jimenez_data/Re4000/parameters.txt') as f:
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