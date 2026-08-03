from pathlib import Path
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def analyze_re(re_name, base_dir, ax):
    folder = base_dir / re_name
    if not folder.exists():
        print(f"Skipping {re_name}: folder not found")
        return

    os.chdir(folder)

    sys.path.insert(0, str(folder))
    sys.modules.pop("bandpass", None)
    sys.modules.pop("flux", None)

    import bandpass as bp
    import flux

    nx, ny, nz, Lx, Ly, Lz, Re_tau, Re = bp.grid()
    nu = 1.0 / Re

    csv_path = folder / "corsin_scale.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        ax.plot(df["y"] * Re_tau, df["L_c"], label=f"$Re_{{\\tau}}=${re_name[2:]}")
        return

    with open(f"{re_name}U", "rb") as fid:
        data = np.fromfile(fid, "float32")
    u = np.reshape(data, (nx, ny, nz), order="F")

    with open(f"{re_name}V", "rb") as fid:
        data = np.fromfile(fid, "float32")
    v = np.reshape(data, (nx, ny, nz), order="F")

    with open(f"{re_name}W", "rb") as fid:
        data = np.fromfile(fid, "float32")
    w = np.reshape(data, (nx, ny, nz), order="F")

    mean_velocity_path = folder / "Um.txt"
    if mean_velocity_path.is_file():
        U = np.atleast_1d(np.loadtxt(mean_velocity_path, ndmin=1)).reshape(-1)
    else:
        raise FileNotFoundError(f"Mean velocity file {mean_velocity_path} not found.")

    y = np.empty(ny)
    if (folder / "ymesh4000.dat").is_file():
        with open(folder / "ymesh4000.dat", "rb") as fid2:
            y = np.fromfile(fid2, ">f")
    elif (folder / "ymesh2000.dat").is_file():
        with open(folder / "ymesh2000.dat", "rb") as fid2:
            y = np.fromfile(fid2, ">f")
    else:
        for i in range(ny):
            y[i] = 1.0 - np.cos(np.pi * i / (ny - 1))

    if U.size != ny:
        if U.size == ny - 1:
            U = np.concatenate(([U[0]], U))
        elif U.size == ny + 1:
            U = U[1:]
        else:
            raise ValueError(f"Unexpected Um.txt length {U.size} for ny={ny}")

    S = np.gradient(U, y)

    ux = flux.gradx(u)
    uy = np.gradient(u, y, axis=1)
    uz = flux.gradz(u)

    vx = flux.gradx(v)
    vy = np.gradient(v, y, axis=1)
    vz = flux.gradz(v)

    wx = flux.gradx(w)
    wy = np.gradient(w, y, axis=1)
    wz = flux.gradz(w)

    diss = (1/Re)*(2*(ux**2+vy**2+wz**2)+(uy+vx)**2+(uz+wx)**2+(vz+wy)**2)
    diss = np.mean(diss, axis=(0, 2))

    safe_S = np.maximum(np.abs(S) ** 3, np.finfo(float).tiny)
    L_c = np.sqrt(diss / safe_S)

    mask = np.arange(ny) <= ny // 2
    y_trim = y[1:ny // 2 + 1]
    L_c_trim = L_c[1:ny // 2 + 1]

    df = pd.DataFrame({"y": y_trim, "L_c": L_c_trim})
    df.to_csv(csv_path, index=False)

    ax.plot(y_trim * Re_tau, L_c_trim, label=f"$Re_{{\\tau}}=${re_name[2:]}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir
    if not any((base_dir / re_name).exists() for re_name in ["Re550", "Re950", "Re2000", "Re4000"]):
        base_dir = script_dir.parent

    fig, ax = plt.subplots(figsize=(8, 5))
    for re_name in ["Re550", "Re950", "Re2000", "Re4000"]:
        print(f"Analyzing {re_name}...")
        analyze_re(re_name, base_dir, ax)

    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$L_c$")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(base_dir / "corsin_scale_comparison.eps", format="eps")
    fig.savefig(base_dir / "corsin_scale_comparison.png", format="png")
