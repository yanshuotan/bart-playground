# Mixing difficulty and the size–noise scale κ = n/(1−R²)

这里是 information-scale note 和论文里 κ 那部分（§3.5 的粉色句子和附录 B.3 `app:kappa`、表 `tab:kappa`）用到的脚本、中间结果和 note 源文件。

## 论文里的用法

- **定义**：κ = n/(1−R²_CV)。其中 1−R²_CV 是在训练集上做 5 折交叉验证的 HistGradientBoosting（500 次迭代，其余用 sklearn 默认）的 MSE 除以 var(y_train)，再对 5 个 run 取平均。整个计算在采样之前完成，不依赖任何 MCMC 输出。
- **理论依据**：Tan et al. (2024, arXiv:2406.19958) 的 Prop. B.1 指出，不同树结构之间的 BIC 差按 n/σ² 增长。把它除以响应变量的方差，就得到 κ。
- **结果**（论文中的 12 个 setting）：
  - κ 与 production Default 中位 R-hat 的 Spearman 相关为 0.96，与 thinned long record 为 0.86，与 MTMH+PT 为 0.90。
  - 只用 n：分别是 0.65、0.60、0.65。只用 1/(1−R²)：分别是 0.43、0.40、0.44。
  - κ > 2.5×10⁴ 的 4 个 setting（Superconduct、SeoulBike、CCPP、CPU activity），正好就是 thinned long 和 MTMH+PT 的中位 R-hat 都还大于 1.1 的那 4 个。
- **局限**（附录里也写了）：
  - κ 只能排序，不能校准。Friedman、Airfoil、CalHousing 的 κ 几乎相同，但 R-hat 在 1.23 到 1.30 之间。
  - 只有 12 个 setting。
  - CV 噪声比例取决于所用的学习器。在 DART 上它是 0.09–0.12，而后验估计约为 0.037。

复现附录表格和上面的相关系数：

```
python kappa_appendix.py            # 读 kgbm.pkl，R-hat 三列取自论文附录表
```

## 文件

| 文件 | 作用 |
|---|---|
| `kgbm.py` → `kgbm.pkl` | 计算 48 个 setting 的 CV 噪声比例 1−R²_CV：12 个主实验、20 个 size/noise sweep、16 个受控实验。键为 `(group, name)`，值为 `(n, 1−R²_CV)` |
| `kappa_appendix.py` | 复现 `tab:kappa` 和附录中的 Spearman 数值 |
| `xds.py` → `xds_rows.pkl` | 对每个论文 setting（5 个 run）汇总：n、后验 1−R²、R-hat（Default 10k、Default 1e7、MTMH+PT）、链间与链内离散度相对 σ 的大小、在 fixed panel 上 BART 与 HGB/RF 的 RMSE 对比 |
| `master.py` → `master_rows.pkl` | 对 20 个 size/noise sweep setting（`paper_store_20261028/store_scaling`）做同样的汇总 |
| `mixexp.py`, `mixana.py` → `mixana_rows.pkl` | 受控实验：SeoulBike、Airfoil、Abalone 上用真实 y、真实 y 加噪声、f̂(X) 加噪声三种响应；4 条 Default 链 × 10k；本地 8 核约 50 分钟。链的抽样没有保存，只保留汇总 |
| `consol.py` → `table48.pkl` | 合并上面 48 个 setting，计算 Spearman/Kendall |
| `drift.py` → `drift.pkl` | 1e7 Default 长链中 σ 和 test RMSE 按迭代十分位的变化 |
| `acc.py` → `acc_all.pkl` | Default 中 grow/prune/change/swap 的接受率 |
| `fig.py` → `mixing_vs_information.png` | 早期探索用的图 |
| `fig2.py` → `note/figures/rhat_vs_kappa.png` | note 的 Figure 1 |
| `figlong.py` → `note/figures/long_pt_{worst2,traces}.png` | note 的 Figure 2、3（SeoulBike run 0 的 1M MTMH+PT 长链） |
| `bilingual.py` | 把英文版和中文版按 section 交错，生成 `note/info_scale_bilingual.tex` |
| `note/` | note 的英文版、中文版和中英对照版，均附 tex 和 pdf；在 `note/` 目录下用 xelatex 或 tectonic 编译 |

## 运行依赖

这些是探索性脚本，没有整理成可以一键运行的流水线：

- 大多数脚本的第一个参数 `S` 是工作目录，脚本在里面读写 `*.pkl`。部分脚本还需要：
  - `S/bp`：本仓库的一份副本，用来 import `fixed100_support` 等模块；
  - `S/datasets.pkl`：一个 dict，`name -> (X, y)`，由 `fixed100_support.load_dataset` 生成。
- 论文实验的 store 路径用环境变量 `PAPER_STORE` 指定，默认是 `paper_store_20261028`，即论文用的那份 store。
- `figlong.py` 读取的 `S/longpt/seoulbike_run0_mtmh_pt_summary.npz` 是从 Vanda 上 `/scratch/e1377027/long_pt/fixed100_SeoulBike/` 汇总出来的，原始输出在那里，runner 是 `hpc_opt/long_pt.py`。
- 已经算好的中间结果（`*.pkl`）都放在这里，所以 `kappa_appendix.py`、`consol.py`、`fig2.py` 可以直接重跑。

## 其他发现（note 里有，论文里没有写）

1. 用后验 σ 估计 κ 时，45 个 setting 上的 Spearman 为 0.91。只用 n 时为 0.79，样本是之前已有的 29 个 setting。
2. 受控实验：
   - SeoulBike 上，真实 y 的 R-hat 是 1.53，同等噪声比例下设定正确的半合成数据是 1.56。噪声比例升到 0.4 后，两者分别降到 1.21 和 1.20。
   - Airfoil 加噪声后 R-hat 从 1.40 降到 1.02。
   - 把 Abalone 改成高信噪比（1−R² 从 0.38 降到 0.01）后，R-hat 反而从 1.08 升到 1.29。
3. 难的 setting 上，1e7 的 Default 长链还在漂移：SeoulBike 的 σ 在整个运行中下降了 16%，而且还在降，但 test RMSE 没有改善。1M 的 MTMH+PT 长链也有同样的漂移，树也在持续变大。
4. 1M 的 MTMH+PT 和 1e7 的 Default，按论文 §6.3 的成本单位计算量相当：SeoulBike 上分别约 840 和 1000 个单位。
