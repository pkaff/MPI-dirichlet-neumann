from mpi4py import MPI
import numpy as NP
import sys
import matplotlib.pylab as plt
from scipy.linalg import block_diag
from numpy.linalg import inv

#variables used
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nIterations = 10
omega = 0.8
dx_inv = 20 #dx = 1/dx_inv, dx_inv has to be integer
u_old = 0 #will be used to store old grid matrix for relaxation

#initializing arrays
dir1 = NP.ones([dx_inv - 1]) #dirichlet left
dir2 = NP.ones([dx_inv - 1]) #dirichlet right
neu1 = NP.ones([dx_inv - 1]) #neumann left
neu2 = NP.ones([dx_inv - 1]) #neumann right

#big room
if (rank == 1):
    #create matrix A in Au = b, dim defined differently for different processes
    dim = (dx_inv-1)*(2*dx_inv-1) #initialize dimension of the room
    B = -4*NP.diag(NP.ones(dx_inv - 1)) + NP.diag(NP.ones(dx_inv - 2), 1) + NP.diag(NP.ones(dx_inv - 2), -1) #block matrix on the diagonal, -4 on main diagonal and ones otherwise
    B = [B] * (2*dx_inv - 1)
    A = block_diag(*B)
    sub = NP.ones(dim - (dx_inv - 1))
    A = A + NP.diag(sub, dx_inv - 1) + NP.diag(sub, -(dx_inv - 1))
    #create b matrix in Au = b
    b = NP.zeros([2*dx_inv - 1, dx_inv - 1]) #initialize boundary conditions
    for i in range(2*dx_inv - 1):
        b[i, 0] += -15 #normal wall, gammas initialized to 15 too.
        b[i, dx_inv - 2] += -15

    for i in range(dx_inv - 1):
        b[0, i] += -40 #gamma H
        b[2*dx_inv - 2, i] += -5 #gamma WF
    
#small rooms
elif (rank == 0 or rank == 2):
    comm.Send(dir1, dest=1)
    #create matrix A in Au = b, dim defined differently for different processes
    dim = (dx_inv-1)*(dx_inv) #dim of the room
    B = -4*NP.diag(NP.ones(dx_inv)) + NP.diag(NP.ones(dx_inv - 1), 1) + NP.diag(NP.ones(dx_inv - 1), -1) #block matrix on the diagonal, -4 on main diagonal and ones otherwise
    B[-1, -1] = -3 #last element is -3 due to neumann conditions
    B = [B] * (dx_inv - 1)
    A = block_diag(*B)
    sub = NP.ones(dim - dx_inv)
    A = A + NP.diag(sub, dx_inv) + NP.diag(sub, -dx_inv)
    #create b matrix in Au = b
    b = NP.zeros([dx_inv - 1, dx_inv]) #small room includes gridpoints on the walls

    for i in range(dx_inv-1):    
        b[i - 1, 0] += -40 #heated wall

    for i in range(dx_inv):
        b[0, i] += -15 #top wall
        b[dx_inv - 2, i] += -15 #bottom wall
        #no need to initialize the gamma wall since it will be overwritten, and it is now 0 which is fine

#terminate all other processes than the 3 necessary
if (rank > 2):
    sys.exit()

A = inv(A) #invert A for solve later

#functions used in loop
#function calculating heat in a big room with two dirichlet conditions
def Omega2(dir1, dir2):
    #u_old is global
    global u_old
    dir1[0] += 15 # define corners to be of degree 15
    dir1[-1] += 15
    dir2[0] += 15
    dir2[-1] += 15
    #update boundary conditions
    for i in range(len(dir1)):
        b[i + dx_inv - 1, 0] = -dir1[i]
        b[i, dx_inv - 2] = -dir2[i]
        
    #give b 1x(dx_inv - 1) dimension
    b_flat = b.flatten()

    #solve Au = b
    #u = NP.linalg.solve(A, b_flat)
    u = A.dot(b_flat)

    #relax
    u = omega * u + (1 - omega) * u_old

    #save u_old
    u_old = u

    neu1 = dir1 #for dimensions
    neu2 = dir2
    #calculate each neumann condition along two walls
    for i in range(1, len(dir1) + 1):
        neu1[-i] = 1./dx_inv**2 * (u[-(i * (dx_inv-1))] - dir1[-i])
        neu2[i - 1] = 1./dx_inv**2 * ([i * (dx_inv - 1) - 1] - dir2[i - 1])
    return [neu1, neu2]
    

#function calculating heat in a small room with one neumann condition
def Omega13(neu):
    #u_old is global
    global u_old
    neu[0] += 15 #define corners to be of degree 15
    neu[-1] += 15
    #update boundaries
    for i in range(len(neu)):
        b[i, -1] = -neu[i]    

    #flatten
    b_flat = b.flatten()

    #solve Au = b
    #u = NP.linalg.solve(A, b_flat)
    u = A.dot(b_flat)

    #relax
    u = omega * u + (1 - omega) * u_old

    #save u_old
    u_old = u

    diri = neu #for dimensions
    #calculate dirichlet condition, taking only the values along the wall
    for i in range(1, len(neu) + 1):
        diri[i - 1] = u[i * dx_inv - 1]

    return diri


#start of loop //MAIN//
for i in range(nIterations):
    #Left room
    if (rank == 0):
        #receive solutions for boundary
        comm.Recv(neu1, source=1)

        #solve
        dir1 = Omega13(neu1)

        #send
        comm.Send(dir1, dest=1)
    #Center room
    elif (rank == 1):
        #receive the solutions for boundary
        comm.Recv(dir1, source=0)
        comm.Recv(dir2, source=2)

        #Solve for this room
        [neu1, neu2] = Omega2(dir1, dir2)

        #Send results
        comm.Send(neu1, dest=0)
        comm.Send(neu2, dest=2)
    #Right room
    elif (rank == 2):
        #receive
        comm.Recv(neu2, source=1)

        #solve
        dir2 = Omega13(neu2)

        #send
        comm.Send(dir2, dest=1)

if (rank == 1):
    u_old = NP.reshape(u_old, (-1, dx_inv - 1))
else:
    u_old = NP.reshape(u_old, (-1, dx_inv))
    #mirror right room
    if (rank == 2):
        u_old = NP.fliplr(u_old)

if (rank == 0):
    #send rows to big room
    for i in range(dx_inv - 1):
        comm.Send(NP.array(u_old[i,:]), dest=1, tag=i)
if (rank == 2):
    #send rows to big room
    for i in range(dx_inv - 1):
        comm.Send(NP.array(u_old[i,:]), dest=1, tag=i)
if (rank == 1):
    #pad
    u_p = NP.pad(u_old, ((0, 0), (dx_inv, dx_inv)), 'constant')

    u_l = NP.zeros((dx_inv - 1, dx_inv))
    u_r = NP.zeros((dx_inv - 1, dx_inv))
    ul = NP.zeros(dx_inv)
    ur = NP.zeros(dx_inv)
    #receive rows of left and right room matrices
    for i in range(dx_inv - 1):
        comm.Recv(ul, source=0, tag=i)
        comm.Recv(ur, source=2, tag=i)
        u_l[i, :] = ul
        u_r[i, :] = ur

    #pad left and right room
    u_l_p = NP.pad(u_l, ((dx_inv, 0), (0, dx_inv + (dx_inv - 1))), 'constant')
    u_r_p = NP.pad(u_r, ((0, dx_inv), (dx_inv + (dx_inv - 1), 0)), 'constant')

    #print(u_l_p)
    #print(u_r_p)
    u_plt = u_p + u_l_p + u_r_p
    #plot
    fig = plt.imshow(u_plt)
    colorbar = plt.colorbar()
    colorbar.set_label('temperature')
    plt.show()
    
