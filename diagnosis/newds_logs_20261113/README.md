# video_transcoding 短链失败报告（job 1548166，exit=1）

## 当前状态（2026-11-21 10:30）

除 video_transcoding 外，所有数据集的短链、ext、长链都已 exit=0（见文末「同批其他任务」）。video_transcoding：

| 部分 | job | 状态 |
|---|---|---|
| ext | 1552287 | 完成，exit=0 |
| 短链 run0–3 | 1552237 | 完成，4 个方法的结果都已写入 store。job 本身 exit=1，原因是 run4 被 OOM 杀掉 |
| 短链 run4 补跑 | 1593169 | 排队中，等长链 1548182 结束后开始，预计约 21 小时 |
| 长链 run1、run3 | 1548182 | 完成，已写入 store |
| 长链 run0、run4 | 1548182 | 各剩 1 条链：9830/10000、8490/10000 行，预计 11-21 晚上结束；内存 329 GB |
| 长链 run2 补跑 | 1593168 | 11-20 01:50 开始，4 条链 3950/6080/4000/5750 行，约 120–185 行/小时，预计 11-23 中午前后结束；内存 98 GB（申请 150 GB） |

细节见下面「2026-11-19 更新」。

- 脚本：`newds_short.pbs`，原样未改；commit `0612bf8`
- 节点：IN-001，20 核，`mem=200gb`
- 时间：2026-11-13 08:25 开始，12:55 结束（墙钟 4:30）
- 数据：`video_transcoding` subsample 到 10000 行，p = 24，train n = 7425，fixed test n = 100
- PBS 输出的最后一行：`cgroup/OOM: Killed because of memory limit`
- 峰值内存：`resources_used.mem` = 209 GB，`vmem` = 223 GB，超过了申请的 200 GB

## 每个 run 的情况

每条链依次跑 default → mtmh → default_pt → mtmh_pt。
5 个 run 的 4 条链都跑完了 default（约 2.5 分钟）和 mtmh（约 12 分钟），全部失败在 **default_pt** 阶段：

| run | 梯子温度数 | 失败位置 | 错误 |
|---|---|---|---|
| run0 | 79 | default_pt 7300/10000（约 2.5 小时） | `LinAlgError: Matrix is not positive definite` |
| run1 | 78 | default_pt 101/10000（刚开始） | `TerminatedWorkerError`，SIGKILL，即 OOM |
| run2 | 80 | default_pt 8500/10000（约 3 小时） | `LinAlgError: Matrix is not positive definite` |
| run3 | 80 | default_pt 9900/10000（约 3.4 小时） | `LinAlgError: Matrix is not positive definite` |
| run4 | 78 | 其他链进入 default_pt 时被杀，当时这条链的 mtmh 在 8610/10000 | `TerminatedWorkerError`，SIGKILL，即 OOM |

`/scratch/e1377027/bart_store_newds/fixed100_VideoTranscoding_subsample10000/` 下只有 `metadata`，没有写出任何 run 的结果。

## 两个问题

### 1. 内存（run1、run4）

- 这个数据集的梯子搜索选出约 80 个温度（78–80）。
- 20 个进程（5 个 run × 4 条链）几乎同时进入 default_pt，总内存超过 200 GB，被 cgroup 杀掉。
- run1 和 run4 退出后内存降下来，其余 3 个 run 才得以继续跑。

### 2. Cholesky 失败（run0、run2、run3）

三个 run 的调用栈相同：

```
bart.py:1592  _fit_loop_serial
bart.py:1481  _swap_step_serial_batched
bart.py:1391  _refresh_state_tempered_params
priors.py:268 resample_leaf_vals → _resample_leaf_vals_numba
numba/np/linalg.py cho_impl → LinAlgError: Matrix is not positive definite
```

- 发生在 PT 交换温度之后、重抽叶子值的时候。
- 每次都在跑了几千次迭代以后，属于偶发的数值问题。
- run0 的梯子搜索阶段还出现过一条警告：`float32 SVD of the (1000, 250) leaf basis failed ... retrying in float64`。

## 补充：video_transcoding 的 ext 也内存超限（job 1548174，exit=1）

- ext 只跑 Default，不用 PT，所以和温度数无关。
- 申请 40 GB，`resources_used.mem` = 41.9 GB。PBS 输出：`cgroup/OOM: Killed because of memory limit`。
- 被杀时只跑了 6 分钟，第一块（10,000 次迭代）还没写盘，所以真实峰值大于 42 GB。

各数据集 ext 的实际内存占用：

| 数据集 | 训练集 n | 内存 | 墙钟 |
|---|---|---|---|
| cars | ~600 | 4.6 GB | 0:09 |
| qsar_fish | ~680 | 6.8 GB | 0:12 |
| space_ga | ~2330 | 6.9 GB | 0:11 |
| cps88wages | 7425 | 16.0 GB | 0:19 |
| black_friday | 7425 | 17.2 GB | 0:22 |
| **video_transcoding** | **7425** | **> 41.9 GB（被杀）** | 0:06 |

我们的推测（没有直接统计树深）：

- 训练集大小一样，video_transcoding 的内存却是 cps88wages、black_friday 的 2.5 倍以上，所以差别出在每个状态的大小上。
- `Tree` 的数组是按堆下标存储的（子节点为 2i+1、2i+2），长度随树深加倍。每一块要保存 10,000 个状态，每个状态 100 棵树。
- video_transcoding 的噪声很小，树长得深，单个状态就大。这和短链需要约 80 个温度是一致的。

## 我们这边的调整（只改了内存申请，其他参数没动）

| 任务 | 原来 | 现在 |
|---|---|---|
| video_transcoding 短链重跑（job 1552237） | 200 GB | 300 GB（按你的命令） |
| video_transcoding ext 重跑（job 1552287） | 40 GB | **160 GB** → 已完成，exit=0，峰值 53.6 GB，墙钟 0:30 |
| video_transcoding 长链（job 1548182，排队中，`qalter`） | 200 GB | **330 GB** |

- 长链每一块也会先在内存里保留完整的 10,000 个状态，再抽稀写盘，所以每个 run 的峰值和 ext 差不多。5 个 run 同时跑。
- 已完成或正在跑的 5 个长链，内存都正好是各自 ext 的 4.8–5.0 倍：

  | 数据集 | ext 内存 | 长链内存 | 倍数 |
  |---|---|---|---|
  | cars | 4.6 GB | 21.9 GB | 4.8 |
  | qsar_fish | 6.8 GB | 33.2 GB | 4.9 |
  | space_ga | 6.9 GB | 34.4 GB | 5.0 |
  | cps88wages | 16.0 GB | 78.1 GB | 4.9 |
  | black_friday | 17.2 GB | 84.4 GB | 4.9 |

- 所以 video_transcoding 长链大约需要 5 × 53.6 ≈ 268 GB，留约 20% 余量后申请 330 GB。最初改到的是 400 GB，ext 测出峰值后调到了 330 GB。原来的 200 GB 显然不够。

## 2026-11-19 更新：长链和短链重跑各有一个 run 内存超限

| 任务 | 申请 | 被杀的 run | 时间 | 其余 run |
|---|---|---|---|---|
| 长链 1548182 | 330 GB | run2，`TerminatedWorkerError`（SIGKILL），只存了 90–130 行 | 11-18 07:36（开跑 51 分钟） | run0/1/3/4 正常，4040–6620/10000 行，约 145 行/小时；内存 329.3 GB，贴着上限 |
| 短链重跑 1552237 | 300 GB | run4，同上 | 11-19 13:41，PT 阶段 | run0–3 正常，在最后一个 PT 方法的 43–46%；内存 318 GB |

- 长链 run2 不是卡住，是被杀了。每个 run 实际约 82 GB，比 ext 峰值 53.6 GB 大约 50%，上面按 5 × ext 估算偏低。推测是长链越跑树越深，状态越大。
- 短链的温度上限 60 已生效（5 个 run 的梯子都是 60 个温度），但 5 个 run 同时跑仍然超过 300 GB。
- 一个 run 被杀只影响它自己，其余 run 继续跑，两个任务都没有停。
- 每个 run 跑完就单独写结果（长链按 `run00X` 分目录，短链按 run 保存），补跑不会覆盖其他 run。

### 补跑（只补被杀的 run，参数不变）

- 两个 pbs 脚本只改了一行：`for r in 0 1 2 3 4` → `for r in ${RUNS:-0 1 2 3 4}`。不传 `RUNS` 时和原来完全一样。原版备份在 `/scratch/e1377027/newds_{short,long}.pbs.orig`。
- 单个 run 只有 4 个进程，所以申请 4 核。

| 任务 | job | 资源 | 命令 |
|---|---|---|---|
| 长链 run2 | 1593168 | 4 核，150 GB | `qsub -q auto_free -N newds_long_r2 -l select=1:ncpus=4:mpiprocs=1:ompthreads=1:mem=150gb -v DATASET=video_transcoding,COMMIT=d2b2b2a,RUNS=2 newds_long.pbs` |
| 短链 run4 | 1593169 | 4 核，200 GB | `qsub -q auto_free -N newds_short_r4 -l select=1:ncpus=4:mpiprocs=1:ompthreads=1:mem=200gb -v DATASET=video_transcoding,COMMIT=d2b2b2a,RUNS=4 newds_short.pbs` |

- 免费队列每人最多同时跑 2 个任务。短链重跑 11-20 01:50 结束后，长链 run2 补跑随即开始；短链 run4 补跑要等长链 1548182 结束。
- 补跑会覆盖 `run2.log` / `run4.log`，被杀时的日志已另存为 `run2_oom_1548182.log`、`run4_oom_1552237.log`。

## 同批其他任务

截至 2026-11-21，以下全部 exit=0：

| 数据集 | 短链 | ext | 长链 |
|---|---|---|---|
| space_ga | 1548162 | 1548170 | 1548178 |
| qsar_fish | 1548164 | 1548172 | 1548180 |
| cars | 1548165 | 1548173 | 1548181 |
| cps88wages | 1548167 | 1548175 | 1548183 |
| black_friday | 1548168 | 1548176 | 1548184 |
| fiat500 | 1552235 | 1552238 | 1552240 |
| kin8nm | 1552236 | 1552239 | 1552241 |

- black_friday 的短链开始于 pull 之前，跑的是旧代码，梯子没有 60 的上限，峰值内存 162 GB。
- 已取消：munich_rent、buzz_twitter 的全部任务。它们的结果和日志已删除；munich_rent 跑完的短链按你的要求作废。
- 新提交（commit `d2b2b2a`）：fiat500、kin8nm 的短链、ext、长链。下载后 shape 和 md5 已核对，都和你给的一致。
- 注意：pull 之前提交、排队中的 ext 和长链，日志里记的是 `commit=0612bf8`，实际执行的是 `d2b2b2a`。新提交只改了 PT 相关的代码，所以对 Default 没有影响。

## 文件

- `video_transcoding_condensed/`：精简版日志。去掉了进度条，只保留每个进度条的完成行和出错前的最后一行，其余内容（梯子信息、警告、完整 traceback）原样保留。另外包含 driver log 和 PBS 输出 `newds_short.o1548166`。
- `video_transcoding_logs.tgz`：job 1548166 的原始日志（driver、run0–4、PBS 输出），原样未改。
