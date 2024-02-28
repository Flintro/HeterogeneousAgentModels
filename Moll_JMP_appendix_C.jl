# Direct translation of Ben Moll's codes for his 2014 paper into Julia.

using Distributions
using Plots
using LinearAlgebra


# PARAMETERS
r = 0.05
d = 0.05
al = 1/3
Corr = 0.85
nu = -log(Corr)
sig = 0.56
sig2 = sig^2
Var = sig2 / (2*nu)

la = 1.2

# GRID PARAMETERS
I = 500
zmin = 0
# zmax = 95-percentile of log-normal distribution:
z_stan_95 = 1.6449
logzmax = z_stan_95*sqrt(Var)
zmax = exp(logzmax)
check = cdf(LogNormal(0, sqrt(Var)), zmax) # compute log-normal CDF
zz = range(zmin, zmax, length=I)
h = (zmax - zmin) / (I - 1)
h2 = h^2
ZFB = zmax^al # FIRST-BEST TFP

T = 100 # TIME LENGTH
N = 500 # NUMBER OF TIME STEPS
dt = T / (N - 1) # TIME STEP

# INITIAL WEALTH SHARES = STATIONARY DIST. OF OU WITH BARRIER
w0 = pdf.(LogNormal(0, sqrt(Var)), zz)
w0 ./= sum(h * w0)

# START FROM THE STEADY STATE CORRESPONDING TO w0
int = zeros(I-1)
for i = 1:I-1
    int[i] = h * sum(w0[i+1:I])
end
ibar0 = findfirst(x -> x < 1, la .* int) # Julia's findfirst returns the index directly
f0 = 1/la - int[ibar0]
E0 = la * (zz[ibar0] * f0 + h * sum(zz[ibar0+1:I] .* w0[ibar0+1:I]))
Z0 = E0^al # truncated expectation
K0 = (al * Z0 / (r + d))^(1 / (1 - al))

# ORNSTEIN-UHLENBECK PROCESS
mu = -nu .* zz .* log.(zz) + sig2 / 2 .* zz
s2 = sig2 .* zz.^2

a = nu .+ nu .* log.(zz) .+ sig2 / 2
b = -mu + 2 * sig2 .* zz
c = 1/2 * sig2 .* zz.^2

x = b * dt / (2 * h) - c * dt / h2
y = 1 .- a * dt + c * 2 * dt / h2
z = -b * dt / (2 * h) - c * dt / h2

B = zeros(I, I)
A = zeros(I-1, I-1)

for i = 2:I-1
    B[i, i-1] = x[i]
    B[i, i] = y[i]
    B[i, i+1] = z[i]
end
A1 = B[2:I-1, 2:I]
A = [A1; h * ones(1, I-1)]

K = zeros(N+1)
KFB = zeros(N+1)
YFB = zeros(N+1)
w = zeros(I, N+1)

K[1] = K0
KFB[1] = K0
w[:, 1] = w0
# Initialize arrays for storing computations
ibar_n = zeros(Int, N)
zbar = zeros(N)
f = zeros(N)
mkt = zeros(N)
E = zeros(N)
Z = zeros(N)
Y = zeros(N)
W = zeros(N)
ze = zeros(N)
R = zeros(N)
pi = zeros(N)
gK = zeros(N)
sav = zeros(I, N) # Note: Adjusted dimensions for sav to store per each n
wsum = zeros(N)
# COMPUTE TRANSITION DYNAMICS
for n = 1:N
    int = zeros(I-1)
    for i = 1:I-1
        int[i] = h * sum(w[i+1:I, n])
    end
    if la == 1
        ibar = 1
    else
        ibar = findfirst(x -> x < 1, la .* int)
        ibar = isnothing(ibar) ? I : ibar # Handling case where no value satisfies the condition
    end
    ibar_n[n] = ibar
    zbar[n] = zz[ibar]
    f[n] = 1/la - int[ibar]
    mkt[n] = la * (f[n] + h * sum(w[ibar+1:I, n])) - 1
    E[n] = la * (zz[ibar] * f[n] + h * sum(zz[ibar+1:I] .* w[ibar+1:I, n]))
    Z[n] = E[n]^al
    Y[n] = Z[n] * K[n]^al
    W[n] = (1-al) * Z[n] * K[n]^al
    ze[n] = zbar[n] / E[n]
    R[n] = al * ze[n] * Z[n] * K[n]^(al-1) - d
    pi[n] = al * ((1-al) / W[n])^((1-al) / al)
    gK[n] = al * Y[n] / K[n] - (r+d)
    for i = 1:I
        sav[i, n] = la * max(zz[i] * pi[n] - R[n] - d, 0) + R[n] - r
    end
    Bsav = diagm(sav[:, n] .- gK[n])
    Asav = [Bsav[2:I-1, 2:I]; zeros(1, I-1)]
    AA = A - dt .* Asav
    wn = [w[2:I-1, n]; 1]
    wn1 = AA \ wn
    w[2:I, n+1] = wn1
    w[1, n+1] = 0
    wsum[n] = sum(h .* w[:, n])
    K[n+1] = dt * (al * Y[n] - (r+d) * K[n]) + K[n]
end

# Plotting the results
timeVector = range(0, stop=T, length=length(K))
plot(timeVector, K, linewidth=2, label="Capital Stock (K)")
xlabel!("Time")
ylabel!("Capital Stock (K)")
title!("Evolution of Capital Stock Over Time")