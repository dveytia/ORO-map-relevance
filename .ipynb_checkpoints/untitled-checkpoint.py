import mpi4py
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("Rank:", rank)
print("Size:", size)

print(mpi4py.get_config())