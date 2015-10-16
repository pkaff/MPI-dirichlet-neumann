import matplotlib.pylab as plt
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 1:
    A = np.array([[1., 2, 3, 3], [4., 5, 6, 6], [1., 2, 3, 3]])
    comm.Send(A, dest=0)
else:
    A = np.zeros((3, 4))
    comm.Recv(A, source=1)
    print(A)

