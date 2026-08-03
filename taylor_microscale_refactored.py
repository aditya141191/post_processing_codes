from pathlib import Path
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def analyze_re(re_name, base_dir, ax_lambda, ax_re_lambda):
    if (base_dir / f"{re_name}_taylor_microscale.csv").exists():
        df = pd.read_csv(base_dir / f"{re_name}_taylor_microscale.csv")
        ax_lambda.plot(df["y"] * df["re_tay"], df["tay_micro"] * df["re_tay"], label=f"$Re_{{\\tau}}=${re_name[2:]}")
        ax_re_lambda.plot(df["y"] * df["re_tay"], df["re_tay"], label=f"$Re_{{\\tau}}=${re_name[2:]}")
        return
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

    with open(f"{re_name}U", "rb") as fid:
        data = np.fromfile(fid, "float32")
    u = np.reshape(data, (nx, ny, nz), order="F")

    with open(f"{re_name}V", "rb") as fid:
        data = np.fromfile(fid, "float32")
    v = np.reshape(data, (nx, ny, nz), order="F")

    with open(f"{re_name}W", "rb") as fid:
        data = np.fromfile(fid, "float32")
    w = np.reshape(data, (nx, ny, nz), order="F")

    q = np.mean(u**2 + v**2 + w**2, axis=(0, 2))

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

    ux = flux.gradx(u)
    # uy = flux.grady(u, y, comm)
    uy = np.gradient(u, y, axis=1)
    uz = flux.gradz(u)

    vx = flux.gradx(v)
    # vy = flux.grady(v, y, comm)
    vy = np.gradient(v, y, axis=1)
    vz = flux.gradz(v)

    wx = flux.gradx(w)
    # wy = flux.grady(w, y, comm)
    wy = np.gradient(w, y, axis=1)
    wz = flux.gradz(w)

    diss = (1.0 / Re) * (
        2 * ux**2 + vx**2 + wx**2 + 2 * vy**2 + wy**2 + uz**2 + vz**2 + wz**2 + 2 * (uy * vx + uz * wx + vz * wy)
    )
    diss = np.mean(diss, axis=(0, 2))

    tay_micro = np.sqrt(5.0 * nu * q / diss)
    re_tay = q * np.sqrt(5.0 / (3.0 * nu * diss))

    df = pd.DataFrame({"y": y[1:ny//2+1], "tay_micro": tay_micro[1:ny//2+1], "re_tay": re_tay[1:ny//2+1]})
    df.to_csv(base_dir / f"{re_name}_taylor_microscale.csv", index=False)

    ax_lambda.plot(y[1:ny//2+1] * Re_tau, tay_micro[1:ny//2+1] * Re_tau, label=f"$Re_{{\\tau}}=${re_name[2:]}")
    ax_re_lambda.plot(y[1:ny//2+1] * Re_tau, re_tay[1:ny//2+1], label=f"$Re_{{\\tau}}=${re_name[2:]}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir
    if not any((base_dir / re_name).exists() for re_name in ["Re550", "Re950", "Re2000", "Re4000"]):
        base_dir = script_dir.parent

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for re_name in ["Re550", "Re950", "Re2000", "Re4000"]:
        analyze_re(re_name, base_dir, ax[0], ax[1])

    ax[0].set_xlabel(r"$y^+$")
    ax[0].set_ylabel(r"$\lambda_{Taylor}^+$")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_xlabel(r"$y^+$")
    ax[1].set_ylabel(r"$Re_{\lambda}$")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(base_dir / "taylor_microscale_comparison.eps", format="eps")
    fig.savefig(base_dir / "taylor_microscale_comparison.png", format="png")
    # plt.show()
