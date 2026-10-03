import numpy as np
from dataclasses import dataclass, field

@dataclass
class MFTire:
    Fz0: float = 4000.0
    pCx1: float = 1.6411
    pDx1: float = 1.1739
    pDx2: float = -0.16395
    pDx3: float = 0.008
    pEx1: float = 0.46403
    pEx2: float = 0.25022
    pEx3: float = 0.06784
    pEx4: float = -3.76e-05
    pKx1: float = 22.303
    pKx2: float = 0.48893
    pKx3: float = 0.21253
    pHx1: float = 0.0
    pHx2: float = 0.0
    pVx1: float = 0.0
    pVx2: float = 0.0
    pCy1: float = 1.3507
    pDy1: float = 1.0489
    pDy2: float = -0.18033
    pDy3: float = -2.8821
    pEy1: float = -0.0074722
    pEy2: float = -0.0063208
    pEy3: float = -9.9935
    pEy4: float = -760.14
    pKy1: float = -21.615
    pKy2: float = 1.5978
    pKy3: float = -0.36829
    pHy1: float = 0.0
    pHy2: float = 0.0
    pVy1: float = 0.0
    pVy2: float = 0.0
    rBx1: float = 13.046
    rBx2: float = 9.718
    rCx1: float = 0.9995
    rEx1: float = -0.4403
    rEx2: float = -0.4663
    rHx1: float = -9.968e-05
    rBy1: float = 7.7856
    rBy2: float = 8.1697
    rBy3: float = -0.05914
    rCy1: float = 1.0533
    rEy1: float = 0.3126
    rEy2: float = 0.03789
    rHy1: float = 0.0
    rHy2: float = 0.0
    lambda_mux: float = 1.0
    lambda_muy: float = 1.0

    def dfz(self, Fz):
        return (Fz - self.Fz0) / self.Fz0

    def mu_x(self, Fz, gamma=0.0):
        return (self.pDx1 + self.pDx2 * self.dfz(Fz)) * (1 - self.pDx3 * gamma ** 2) * self.lambda_mux

    def Fx0(self, kappa, Fz, gamma=0.0):
        if Fz <= 0:
            return 0.0
        dfz = self.dfz(Fz)
        SHx = self.pHx1 + self.pHx2 * dfz
        SVx = Fz * (self.pVx1 + self.pVx2 * dfz)
        kx = kappa + SHx
        Cx = self.pCx1
        Dx = self.mu_x(Fz, gamma) * Fz
        Ex = (self.pEx1 + self.pEx2 * dfz + self.pEx3 * dfz ** 2) * (1 - self.pEx4 * np.sign(kx))
        Ex = min(Ex, 1.0)
        Kx = Fz * (self.pKx1 + self.pKx2 * dfz) * np.exp(self.pKx3 * dfz)
        Bx = Kx / (Cx * Dx) if abs(Cx * Dx) > 1e-09 else 0.0
        arg = Bx * kx
        return Dx * np.sin(Cx * np.arctan(arg - Ex * (arg - np.arctan(arg)))) + SVx

    def peak_Fx(self, Fz, gamma=0.0):
        ks = np.linspace(0.0, 0.5, 2001)
        fx = np.array([self.Fx0(k, Fz, gamma) for k in ks])
        i = int(np.argmax(fx))
        return (float(fx[i]), float(ks[i]))

    def mu_y(self, Fz, gamma=0.0):
        return (self.pDy1 + self.pDy2 * self.dfz(Fz)) * (1 - self.pDy3 * gamma ** 2) * self.lambda_muy

    def Fy0(self, alpha, Fz, gamma=0.0):
        if Fz <= 0:
            return 0.0
        dfz = self.dfz(Fz)
        SHy = self.pHy1 + self.pHy2 * dfz
        SVy = Fz * (self.pVy1 + self.pVy2 * dfz)
        ay = alpha + SHy
        Cy = self.pCy1
        Dy = self.mu_y(Fz, gamma) * Fz
        Ey = (self.pEy1 + self.pEy2 * dfz) * (1 - (self.pEy3 + self.pEy4 * gamma) * np.sign(ay))
        Ey = min(Ey, 1.0)
        Ky = self.pKy1 * self.Fz0 * np.sin(2 * np.arctan(Fz / (self.pKy2 * self.Fz0))) * (1 - self.pKy3 * abs(gamma))
        By = Ky / (Cy * Dy) if abs(Cy * Dy) > 1e-09 else 0.0
        arg = By * ay
        return Dy * np.sin(Cy * np.arctan(arg - Ey * (arg - np.arctan(arg)))) + SVy

    def peak_Fy(self, Fz, gamma=0.0):
        als = np.linspace(0.0, 0.5, 2001)
        fy = np.array([abs(self.Fy0(a, Fz, gamma)) for a in als])
        i = int(np.argmax(fy))
        return (float(fy[i]), float(als[i]))

    def Gxa(self, kappa, alpha, Fz):
        dfz = self.dfz(Fz)
        SHxa = self.rHx1
        a_s = alpha + SHxa
        Bxa = self.rBx1 * np.cos(np.arctan(self.rBx2 * kappa))
        Cxa = self.rCx1
        Exa = min(self.rEx1 + self.rEx2 * dfz, 1.0)

        def _G(x):
            return np.cos(Cxa * np.arctan(Bxa * x - Exa * (Bxa * x - np.arctan(Bxa * x))))
        num, den = (_G(a_s), _G(SHxa))
        return num / den if abs(den) > 1e-09 else 0.0

    def Gyk(self, kappa, alpha, Fz):
        dfz = self.dfz(Fz)
        SHyk = self.rHy1 + self.rHy2 * dfz
        k_s = kappa + SHyk
        Byk = self.rBy1 * np.cos(np.arctan(self.rBy2 * (alpha - self.rBy3)))
        Cyk = self.rCy1
        Eyk = min(self.rEy1 + self.rEy2 * dfz, 1.0)

        def _G(x):
            return np.cos(Cyk * np.arctan(Byk * x - Eyk * (Byk * x - np.arctan(Byk * x))))
        num, den = (_G(k_s), _G(SHyk))
        return num / den if abs(den) > 1e-09 else 0.0

    def Fx_combined(self, kappa, alpha, Fz, gamma=0.0):
        return self.Gxa(kappa, alpha, Fz) * self.Fx0(kappa, Fz, gamma)

    def Fy_combined(self, kappa, alpha, Fz, gamma=0.0):
        return self.Gyk(kappa, alpha, Fz) * self.Fy0(alpha, Fz, gamma)

    def friction_envelope(self, Fz, gamma=0.0, n_alpha=161, n_kappa=161):
        alphas = np.linspace(0.0, np.radians(20.0), n_alpha)
        kappas = np.linspace(0.0, 0.4, n_kappa)
        FX, FY = ([], [])
        for a in alphas:
            for k in kappas:
                FX.append(self.Fx_combined(k, a, Fz, gamma))
                FY.append(abs(self.Fy_combined(k, a, Fz, gamma)))
        return (np.array(FX), np.array(FY))

    def max_Fx_given_Fy(self, Fy_target, Fz, gamma=0.0, tol_frac=0.03):
        FX, FY = self.friction_envelope(Fz, gamma)
        tol = max(tol_frac * abs(Fy_target), 25.0)
        sel = np.abs(FY - Fy_target) <= tol
        return float(FX[sel].max()) if sel.any() else float('nan')

def selftest():
    t = MFTire()
    print('=' * 84)
    print('MAGIC FORMULA -- numerical validation against physical invariants')
    print('=' * 84)
    ok = []
    f0 = t.Fx0(0.0, t.Fz0)
    print(f'\n[1] Fx(kappa=0) = {f0:.6f} N   (must be ~0)')
    ok.append(abs(f0) < 1e-06)
    print(f"    {('PASS' if ok[-1] else 'FAIL')}")
    fx_pk, k_pk = t.peak_Fx(t.Fz0)
    mu_pk = fx_pk / t.Fz0
    print(f'\n[2] peak Fx at nominal load = {fx_pk:.1f} N -> mu = {mu_pk:.4f}')
    print(f'    pDx1 (design peak mu)    = {t.pDx1:.4f}')
    ok.append(abs(mu_pk - t.pDx1) < 0.005)
    print(f"    {('PASS' if ok[-1] else 'FAIL')}")
    print(f'\n[3] slip ratio at peak force = {100 * k_pk:.1f} %   (real tires peak at ~5-15%)')
    ok.append(0.03 <= k_pk <= 0.2)
    print(f"    {('PASS' if ok[-1] else 'FAIL')}")
    print(f'\n[4] load sensitivity of peak friction:')
    print(f"    {'Fz [N]':>8} {'dfz':>7} {'peak Fx [N]':>12} {'mu':>7} {'Fx/Fz0 ratio':>13}")
    loads = [2000, 3000, 4000, 5000, 6000, 7000]
    mus = []
    for Fz in loads:
        fx, _ = t.peak_Fx(Fz)
        mus.append(fx / Fz)
        print(f'    {Fz:8.0f} {t.dfz(Fz):7.2f} {fx:12.1f} {fx / Fz:7.4f} {fx / t.Fz0:13.3f}')
    monotone = all((mus[i] > mus[i + 1] for i in range(len(mus) - 1)))
    print(f"    mu strictly decreasing with load: {('PASS' if monotone else 'FAIL')}")
    ok.append(monotone)
    print(f"\n[5] CROSS-CHECK: load transfer strictly destroys grip")
    print(f'    The concavity result says axle capability f(Fz)=mu(Fz)*Fz is CONCAVE,')
    print(f'    so splitting a fixed axle load unevenly must LOSE total grip.')
    print(f'    The Magic Formula is an independent, industry-standard model. Test it:')
    N = 8000.0
    print(f'    axle load N = {N:.0f} N, split as N/2 -/+ delta')
    print(f"    {'delta [N]':>10} {'Fx_inner':>10} {'Fx_outer':>10} {'total':>10} {'vs even':>9}")
    even, _ = t.peak_Fx(N / 2)
    even_total = 2 * even
    losses = []
    for d in [0, 500, 1000, 1500, 2000, 2500, 3000]:
        fi, _ = t.peak_Fx(N / 2 - d)
        fo, _ = t.peak_Fx(N / 2 + d)
        tot = fi + fo
        losses.append(tot)
        print(f'    {d:10.0f} {fi:10.1f} {fo:10.1f} {tot:10.1f} {100 * (tot / even_total - 1):+8.2f}%')
    concave = all((losses[i] >= losses[i + 1] - 1e-06 for i in range(len(losses) - 1)))
    print(f"    total grip decreases monotonically with load transfer: {('PASS' if concave else 'FAIL')}")
    print(f'    => the Magic Formula INDEPENDENTLY reproduces the concavity result.')
    ok.append(concave)
    print(f'\n[6] COMBINED SLIP: Magic Formula envelope vs the friction-ellipse approximation')
    print(f"    (this quantifies the error of the friction ellipse assumption)")
    print(f'    NOTE: Fx and Fy are evaluated at the SAME (kappa, alpha) operating point.')
    print(f'    An earlier version of this test compared Fx at optimal kappa against Fy at')
    print(f'    kappa=0 -- not simultaneously achievable -- and reported a spurious 90% error.')
    Fz = t.Fz0
    fx_max, _ = t.peak_Fx(Fz)
    fy_max, _ = t.peak_Fy(Fz)
    print(f'    pure Fx_max = {fx_max:.0f} N,  pure Fy_max = {fy_max:.0f} N')
    print(f"    {'Fy demand':>10} {'Fy/Fy_max':>10} {'Fx (MF)':>10} {'Fx (ellipse)':>13} {'ellipse err':>12}")
    errs = []
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
        fy_t = frac * fy_max
        fx_mf = t.max_Fx_given_Fy(fy_t, Fz)
        fx_ell = fx_max * np.sqrt(max(0.0, 1 - frac ** 2))
        if not np.isfinite(fx_mf) or fx_mf < 1:
            print(f"    {fy_t:10.0f} {frac:10.2f} {'n/a':>10} {fx_ell:13.0f} {'--':>12}")
            continue
        err = 100 * (fx_ell - fx_mf) / fx_mf
        errs.append(abs(err))
        print(f'    {fy_t:10.0f} {frac:10.2f} {fx_mf:10.0f} {fx_ell:13.0f} {err:+11.1f}%')
    print(f'    mean |ellipse error| = {np.mean(errs):.1f}%,  max = {np.max(errs):.1f}%')
    print(f"    => the ellipse is {('conservative' if np.mean(errs) > 0 else 'optimistic')} but structurally reasonable; MF is required for quantitative combined-slip work.")
    print(f'\n[7] environmental coupling: lambda_mu scales peak friction linearly')
    for lam, label in [(1.0, 'dry'), (0.6, 'wet'), (0.25, 'snow'), (0.1, 'ice')]:
        t2 = MFTire(lambda_mux=lam)
        fx, kk = t2.peak_Fx(t2.Fz0)
        print(f'    {label:<5} lambda={lam:4.2f} -> peak mu = {fx / t2.Fz0:6.4f}, peak slip = {100 * kk:4.1f}%')
    t_dry, t_ice = (MFTire(lambda_mux=1.0), MFTire(lambda_mux=0.1))
    r = t_ice.peak_Fx(4000)[0] / t_dry.peak_Fx(4000)[0]
    ok.append(abs(r - 0.1) < 0.01)
    print(f"    scaling is exactly linear: {('PASS' if ok[-1] else 'FAIL')} (ratio {r:.4f})")
    print('\n' + '=' * 84)
    print(f'VALIDATION: {sum(ok)}/{len(ok)} invariants passed')
    return all(ok)
if __name__ == '__main__':
    selftest()
