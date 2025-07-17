import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


# -----------------------------------------------------------------------------
# Grid Setup
# -----------------------------------------------------------------------------


L = 1
N = 50
dx = L/N

x_data = np.linspace( 0, L, N+1 )


tmax = 2
Nt = 100
dt = tmax/Nt

t_data = np.linspace( 0, tmax, Nt+1 )

X,T = np.meshgrid( x_data, t_data, indexing='ij' )


# -----------------------------------------------------------------------------
# Coefficients and Functions
# -----------------------------------------------------------------------------


mu = np.sin( 2 * np.pi * x_data / L )
dx_mu = 0.5*L / np.pi * np.cos( 2 * np.pi * x_data / L )

sigma = 0.5

D = 0.5*sigma**2


# -----------------------------------------------------------------------------
# System Matrices
# -----------------------------------------------------------------------------

Ix = np.identity(N-1)
ones_x = np.ones( N-1 )

# First order partial derivative (in x)
Px = (  np.diag( ones_x[1:], 1 ) - np.diag( ones_x[1:], -1 )  ) * 0.5/dx

Px[0,0] = -0.5/dx
Px[-1,-1] = 0.5/dx


# Second order partial derivative (in x)
Pxx = (
          - 2*np.diag( ones_x )
          + np.diag( ones_x[1:], 1 )
          + np.diag( ones_x[1:], -1 )
      ) / dx**2

Pxx[0,0] = Pxx[-1,-1] = -1/dx**2


# Main System Matrix and Inverse
E = Ix - dt * ( - mu[1:-1, None] * Px - dx_mu[1:-1, None] * Ix + D * Pxx )
E_inv = np.linalg.inv( E )


# Extended to full length state vectors (vanishing Dirichlet boundary)
M = np.zeros( [N+1]*2 )
M[1:-1, 1:-1] = E_inv



# -----------------------------------------------------------------------------
# Main Simulation Step
# -----------------------------------------------------------------------------


def stepper(p):
    p1 = np.zeros_like(p)
    p1[1:-1] = np.matmul( E_inv, p[1:-1] )
    p1[0], p1[-1] = p1[1], p1[-2]
    return p1


# -----------------------------------------------------------------------------
# Initial data
# -----------------------------------------------------------------------------

fun = 2/L * ( 1 - np.cos( 4 * np.pi * x_data / L ) )

p0 = np.heaviside( L/2 - x_data, 0.5 ) * fun

p = p0


# -----------------------------------------------------------------------------
# Figure Setup
# -----------------------------------------------------------------------------


fig = plt.figure( figsize=[10,8], dpi=80 )
fig.suptitle( 'Probability Density Plot' )
ax = fig.add_subplot()

ax.set_xlabel('x')
ax.set_ylabel('p(x)')

ax.set_xlim( 0, L )
ax.set_ylim( 0, 4/L )

lins, = ax.plot( [], [], color=[0.2, 0.4, 0.8] )


plt.show()


# -----------------------------------------------------------------------------
# Simulation and Animation
# -----------------------------------------------------------------------------


MD = dict( title='', artist='' )
writer = PillowWriter( fps=10, metadata=MD )

with writer.saving( fig, 'Fokker-Planck Sim 1.gif', Nt ):
    for it in range(Nt):
        
        lins.set_data( x_data, p )
        
        p = stepper(p)
        
        writer.grab_frame()







