
import numpy as np
import matplotlib.pyplot as plt

BONES_20 = [
    (0,1),(1,2),(2,3),
    (2,4),(4,5),(5,6),(6,7),
    (2,8),(8,9),(9,10),(10,11),
    (0,12),(12,13),(13,14),(14,15),
    (0,16),(16,17),(17,18),(18,19),
]

def plot_skeleton(seq, frame=32, title="UTD-MHAD skeleton"):
    xyz = seq[frame]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:,0],xyz[:,2],xyz[:,1],s=24)
    for a,b in BONES_20:
        ax.plot([xyz[a,0],xyz[b,0]],[xyz[a,2],xyz[b,2]],[xyz[a,1],xyz[b,1]])
    ax.set_title(title)
    plt.show()

def plot_imu(x, title="UTD-MHAD IMU"):
    plt.figure(figsize=(10,4))
    for c in range(x.shape[1]):
        plt.plot(x[:,c],label=f"ch{c+1}")
    plt.legend(ncol=3)
    plt.title(title)
    plt.xlabel("resampled time")
    plt.show()

def plot_history(df):
    plt.figure(figsize=(8,4))
    plt.plot(df.epoch,df.val_top1,label="val top1")
    plt.plot(df.epoch,df.val_macro_f1,label="val macro-F1")
    plt.legend()
    plt.grid(alpha=.3)
    plt.xlabel("epoch")
    plt.show()

def plot_alignment(A, title="Skeleton -> IMU temporal alignment"):
    plt.figure(figsize=(6,5))
    plt.imshow(np.asarray(A),aspect="auto")
    plt.colorbar()
    plt.xlabel("IMU token")
    plt.ylabel("Skeleton token")
    plt.title(title)
    plt.show()
