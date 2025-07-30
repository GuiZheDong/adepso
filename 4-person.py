import numpy as np
import matplotlib.pyplot as plt
import math

# =====================================
#  1️⃣ 四人博弈的 fitness 和 constraints
# =====================================
def fitness(x):
    # 12个lambda变量
    lam = x[:12]
    lam12, lam21, lam13, lam31, lam14, lam41, lam23, lam32, lam24, lam42, lam34, lam43 = lam

    # 四个玩家的混合策略，每人3个策略概率
    x1_1, x2_1, x3_1 = x[12:15]
    x1_2, x2_2, x3_2 = x[15:18]
    x1_3, x2_3, x3_3 = x[18:21]
    x1_4, x2_4, x3_4 = x[21:24]

    # 四个玩家间的收益矩阵（示例，3x3）
    A = np.array([[-3, 0, 1],
                  [-4, -2, 0],
                  [ 1, -1, -3]])
    B = np.array([[ 0, -3, -1],
                  [-1,  0, -2],
                  [-3, -1,  0]])
    C = np.array([[-2,  1, 0],
                  [ 1, -3, -1],
                  [ 0, -1, -2]])
    D = np.array([[1, -1, 0],
                  [-1, 1, -1],
                  [0, -1, 1]])

    # 计算各玩家间的收益项（示例对应 λ 和矩阵）
    f1 = lam12 + x1_1 * (A[0] @ [x1_2, x2_2, x3_2])
    f2 = lam21 + x2_1 * (A[1] @ [x1_2, x2_2, x3_2])
    f3 = lam13 + x1_1 * (B[0] @ [x1_3, x2_3, x3_3])
    f4 = lam31 + x3_1 * (B[2] @ [x1_3, x2_3, x3_3])
    f5 = lam14 + x1_1 * (D[0] @ [x1_4, x2_4, x3_4])
    f6 = lam41 + x1_4 * (D[0] @ [x1_1, x2_1, x3_1])
    f7 = lam23 + x2_1 * (C[1] @ [x1_3, x2_3, x3_3])
    f8 = lam32 + x3_1 * (C[2] @ [x1_2, x2_2, x3_2])
    f9 = lam24 + x2_1 * (D[1] @ [x1_4, x2_4, x3_4])
    f10 = lam42 + x1_4 * (D[2] @ [x1_2, x2_2, x3_2])
    f11 = lam34 + x3_1 * (D[2] @ [x1_4, x2_4, x3_4])
    f12 = lam43 + x1_4 * (D[2] @ [x1_3, x2_3, x3_3])

    return abs(f1)+abs(f2)+abs(f3)+abs(f4)+abs(f5)+abs(f6)+abs(f7)+abs(f8)+abs(f9)+abs(f10)+abs(f11)+abs(f12)


def constraints(x):
    x1_1, x2_1, x3_1 = x[12:15]
    x1_2, x2_2, x3_2 = x[15:18]
    x1_3, x2_3, x3_3 = x[18:21]
    x1_4, x2_4, x3_4 = x[21:24]

    c1 = abs(x1_1+x2_1+x3_1-1)
    c2 = abs(x1_2+x2_2+x3_2-1)
    c3 = abs(x1_3+x2_3+x3_3-1)
    c4 = abs(x1_4+x2_4+x3_4-1)

    return 1000*(c1+c2+c3+c4)


# =====================================
#  2️⃣ ADEPSO 类
# =====================================
class ADEPSO:
    def __init__(self, fitness, constraints, lower, upper, pop_size, dim, epochs, P):
        self.fitness = fitness
        self.constraints = constraints
        self.lowerbound = lower
        self.upperbound = upper
        self.pop_size = pop_size
        self.dim = dim
        self.epochs = epochs
        self.P = P
        self.population = np.random.rand(pop_size, dim)
        self.fit = np.array([self.fitness(chrom) + self.constraints(chrom) for chrom in self.population])
        self.F0 = 0.4
        self.CR0 = 0.9
        self.NFE = self.pop_size
        self.L = np.full(self.dim, np.inf)
        self.U = np.full(self.dim, -np.inf)
        self.l = np.full(self.dim, np.inf)
        self.u = np.full(self.dim, -np.inf)
        self.best = self.population[np.argmin(self.fit)]
        self.velocities = np.random.rand(pop_size, dim) * 0.1
        self.personal_best = np.copy(self.population)
        self.personal_best_fit = np.array([self.fitness(chrom) + self.constraints(chrom) for chrom in self.personal_best])
        self.global_best = self.best

    def initpop(self):
        self.population = self.lowerbound + (self.upperbound - self.lowerbound) * np.random.rand(self.pop_size, self.dim)
        self.fit = np.array([self.fitness(chrom) + self.constraints(chrom) for chrom in self.population])
        self.best = self.population[np.argmin(self.fit)]

    def mut(self, t):
        lambda_ = 1 - (t / self.epochs)
        F = self.F0 * (2 ** lambda_)
        CR = self.CR0 * (2 ** lambda_)
        mut_population = []
        for i in range(self.pop_size):
            r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
            v = lambda_ * self.population[r1] + F * (self.population[r2] - self.population[r3]) + (1-lambda_) * self.best
            mut_population.append(v)
        return np.array(mut_population), F, CR

    def cross(self, mut_population, CR):
        cross_population = np.copy(self.population)
        for i in range(self.pop_size):
            j = np.random.randint(0, self.dim)
            for k in range(self.dim):
                if np.random.rand() < CR or k == j:
                    cross_population[i][k] = mut_population[i][k]
        return cross_population

    def update_personal_best(self):
        for i in range(self.pop_size):
            total_fit = self.fitness(self.population[i]) + self.constraints(self.population[i])
            if total_fit < self.personal_best_fit[i]:
                self.personal_best[i] = self.population[i]
                self.personal_best_fit[i] = total_fit

    def update_global_best(self):
        min_fit = np.min(self.fit)
        if min_fit < self.fitness(self.global_best) + self.constraints(self.global_best):
            self.global_best = self.population[np.argmin(self.fit)]

    def update_velocity(self, w=0.7, c1=2.5, c2=2.5):
        for i in range(self.pop_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = (w * self.velocities[i] +
                                  c1 * r1 * (self.personal_best[i] - self.population[i]) +
                                  c2 * r2 * (self.global_best - self.population[i]) +
                                  0.1 * np.random.randn(self.dim))

    def update_position(self):
        self.population += self.velocities
        self.population = np.clip(self.population, self.lowerbound, self.upperbound)

    def run(self):
        self.initpop()
        best_hist = []
        for t in range(self.epochs):
            mut_population, F, CR = self.mut(t)
            cross_population = self.cross(mut_population, CR)

            for i in range(self.pop_size):
                off_fit = self.fitness(cross_population[i]) + self.constraints(cross_population[i])
                if off_fit < self.fit[i]:
                    self.fit[i] = off_fit
                    self.population[i] = cross_population[i]
                if off_fit < self.fitness(self.best) + self.constraints(self.best):
                    self.best = cross_population[i]

            self.update_personal_best()
            self.update_global_best()
            self.update_velocity()
            self.update_position()

            best_hist.append(np.min(self.fit))
            print(f"Epoch {t}: Best Fitness = {best_hist[-1]}")
        return self.best, best_hist


# =====================================
#  3️⃣ LRA-CMA-ES 原始实现 (精简封装)
# =====================================
class Solution:
    def __init__(self, dim):
        self.f = float("nan")
        self.x = np.zeros([dim, 1])
        self.z = np.zeros([dim, 1])

def run_LRACMAES(obj_func, dim, mean, sigma, seed=1, max_evals=100000):
    np.random.seed(seed)
    lamb = 4 + int(3 * np.log(dim))
    mu = int(lamb / 2)
    wrh = math.log((lamb + 1.0) / 2.0) - np.log(np.arange(1, mu + 1))
    w = wrh / sum(wrh)
    mueff = 1 / np.sum(w**2)
    cm = 1.0
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    ds = 1 + cs + 2 * max(0.0, np.sqrt((mueff-1.0)/(dim+1.0))-1.0)
    chiN = np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim*dim))

    C = np.eye(dim)
    ps = np.zeros([dim, 1])
    sqrtC = np.eye(dim)
    mean = np.ones([dim, 1]) * mean

    sols = [Solution(dim) for _ in range(lamb)]
    best_hist = []

    evals = 0
    while evals < max_evals:
        for i in range(lamb):
            sols[i].z = np.random.randn(dim, 1)
            sols[i].x = mean + sigma * sqrtC @ sols[i].z
            sols[i].f = obj_func(sols[i].x.flatten())
        evals += lamb
        sols.sort(key=lambda s: s.f)
        best_hist.append(sols[0].f)
        if len(best_hist) >= 300: break

        wz = np.sum([w[i]*sols[i].z for i in range(mu)], axis=0)
        mean += cm * sigma * sqrtC @ wz
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff)*wz
        sigma *= math.exp((cs/ds)*(np.linalg.norm(ps)/chiN - 1))

    return sols[0].x.flatten(), best_hist


# =====================================
#  4️⃣ 主函数：四人博弈对比
# =====================================
if __name__ == "__main__":
    dim = 24
    lower = np.zeros(dim)
    upper = np.ones(dim)

    adepso = ADEPSO(fitness, constraints, lower, upper, pop_size=50, dim=dim, epochs=300, P=0.2)
    best_adepso, hist_adepso = adepso.run()

    best_cmaes, hist_cmaes = run_LRACMAES(lambda x: fitness(x) + constraints(x), dim, mean=0.5, sigma=0.3, seed=1, max_evals=100000)

    plt.plot(hist_adepso, label="ADEPSO")
    plt.plot(hist_cmaes, label="LRA-CMA-ES")
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness")
    plt.title("4-Player Game Optimization")
    plt.legend()
    plt.grid()
    plt.show()

    print("\nADEPSO Optimal solution:", best_adepso)
    print("ADEPSO Optimal value:", fitness(best_adepso) + constraints(best_adepso))
    print("\nLRA-CMA-ES Optimal solution:", best_cmaes)
    print("LRA-CMA-ES Optimal value:", fitness(best_cmaes) + constraints(best_cmaes))
