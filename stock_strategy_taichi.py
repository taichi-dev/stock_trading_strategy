import time
import taichi as ti
ti.init(arch=ti.gpu)


pi = ti.math.pi
nS = 200
nW = 400

T = 24
Sbar = 1.0
rho = 0.99
sigma = 0.01
lamb = 0.01

J = ti.field(dtype=ti.f32, shape=(T + 1, nS + 1, nW + 1))
opt = ti.field(dtype=ti.f32, shape=(T + 1, nS + 1, nW + 1))

p = ti.field(dtype=ti.f32, shape=(nS + 1, nS + 1))


@ti.kernel
def compute_p():  # transition probability of S
    for i in range(nS + 1):
        tot = 0.0
        for j in range(nS + 1):
            Sprev = i / nS * 2
            Snext = j / nS * 2
            Snext_AR = (Sprev - Sbar) * rho + Sbar
            d = Snext - Snext_AR
            pdf = 1 / (sigma * ti.sqrt(2 * pi)) * ti.exp(-0.5 * (d / sigma)**2)
            p[i, j] = pdf
            tot += pdf

        for j in range(nS + 1):  # normalize
            p[i, j] /= tot


@ti.kernel
def compute_Jt(t: ti.i32):
    opt.fill(-1)
    for i, j in ti.ndrange(nS + 1, nW + 1):
        J[t, i, j] = -1e30
        S = i / nS * 2
        for k in range(nW + 1):
            x = (j - k) / nW * 5
            E = 0.0  # expectation
            for l in range(nS + 1):
                val = x * (rho * (S - Sbar) + Sbar - lamb * x) + J[t + 1, l, k]
                E += p[i, l] * val
            if E > J[t, i, j]:
                opt[t, i, j] = x
                J[t, i, j] = E


@ti.kernel
def compute_JT():
    for i, j in ti.ndrange(nS + 1, nW + 1):
        S = i / nS * 2
        x = j / nW * 5 - 2
        val = x * (rho * (S - Sbar) + Sbar - lamb * x)
        J[T, i, j] = val


def main():
    compute_p()
    compute_JT()
    for t in reversed(range(T)):
        compute_Jt(t)


if __name__ == "__main__":
    t0 = time.perf_counter()
    num_tests = 2
    for _ in range(num_tests):
        main()
    t1 = time.perf_counter()
    print(f"Average time elapsed using Taichi: {(t1 - t0) / num_tests}")

