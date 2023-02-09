import taichi as ti
ti.init(arch=ti.gpu)

nS = 200
nW = 400

T = 24
Sbar = 1.0
rho = 0.99
sigma = 0.01
lamb = 0.01

J = ti.field(dtype=float, shape=(T + 1, nS + 1, nW + 1))
opt = ti.field(dtype=float, shape=(T + 1, nS + 1, nW + 1))
p = ti.field(dtype=float, shape=(nS + 1, nS + 1))


@ti.kernel
def compute_p():  # transition probability of S
    for i in range(nS + 1):
        tot = 0.0
        for j in range(nS + 1):
            Sprev = i / nS * 2
            Snext = j / nS * 2
            Snext_AR = (Sprev - Sbar) * rho + Sbar
            epsilon = Snext - Snext_AR
            pdf = 1 / (sigma * ti.sqrt(2 * ti.math.pi)) * ti.exp(-0.5 * (epsilon / sigma)**2)
            p[i, j] = pdf
            tot += pdf

        for j in range(nS + 1):  # normalize
            p[i, j] /= tot


@ti.kernel
def compute_Jt(t: int):
    for i, j in ti.ndrange(nS + 1, nW + 1):
        J[t, i, j] = -1e30
        Si = i / nS * 2
        for k in range(j + 1):
            x = k / nW * 3
            E = 0.0  # expectation
            for l in range(nS + 1):
                val = x * (rho * (Si - Sbar) + Sbar - lamb * x) + J[t + 1, l, j - k]
                E += p[i, l] * val
            if E > J[t, i, j]:
                opt[t, i, j] = x
                J[t, i, j] = E


@ti.kernel
def compute_JT():
    for i, j in ti.ndrange(nS + 1, nW + 1):
        Si = i / nS * 2
        x = j / nW * 3
        val = x * (rho * (Si - Sbar) + Sbar - lamb * x)
        J[T, i, j] = val


def main():
    opt.fill(-1)
    compute_p()
    compute_JT()
    for t in reversed(range(1, T)):
        compute_Jt(t)


if __name__ == "__main__":
    num_tests = 5
    for _ in range(num_tests):
        main()