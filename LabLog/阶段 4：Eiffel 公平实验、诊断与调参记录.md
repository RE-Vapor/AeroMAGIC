# 阶段 4：Eiffel 公平实验、诊断与调参记录

日期：2026-08-13

## 结论边界

本阶段只对 **Eiffel 单场景、起点 0**给出结论。没有运行或标定其余 13 个
Macarons++ 场景，不得把以下 coverage、depth、confidence 或调参结论外推到它们。

代码基线为阶段 3 commit `52c4321859acf399f6ea7353bf12c2abaa09c16a`。主实验固定：

- Eiffel 起点 `start_positions[0] = [2,9,3,4,5]`；
- `random_seed=8`、`torch_seed=9`；
- 6 次观测（pose 0 加 5 次决策）；
- Eiffel 高度标定 `scene_units_per_meter=0.2641176363636364`；
- GT mesh segment collision 开启；相机 pose validity 也读取 GT mesh；
- 两个规划器共用 mapping gathering multiplier `1.0`；
- coverage 始终读取同一 Eiffel GT mesh reference，按 `visibility_ratio` 归一化。

额外先验必须显式披露：SCONE 和 MAGICIAN 的候选 pose validity 与本实验的路径段
collision 都读取 GT mesh；MAGICIAN 还使用 occupancy → imagined Gaussians →
RaDe-GS rasterizer 做 beam search，SCONE 不使用 RaDe-GS。

## 无泄漏设计

在线规划进程只记录 provider depth、`valid_mask & error_mask`、confidence、局部点云、
proxy signed distance、coverage、轨迹、时延和 CUDA memory。在线 JSON 明确写入：

```text
online_only=true
renderer_gt_read=false
renderer_zbuf_role=offline_diagnostic_only
gt_feedback_to_da3=false
```

renderer `zbuf/mask` 只由 `scripts/analyze_planning_diagnostics.py` 在规划子进程退出后
读取，用来计算 depth/geometry/test-fit 诊断。diagnostic JSON 写入
`computed_after_planner_exit=true` 与 `feedback_to_online_planner=false`。GT depth、mask、
test-fit scale/shift 从未回流到 DA3 cache、point cloud、proxy state、候选评分或轨迹。

## 可复现入口

完整矩阵（只生成配置，不占 GPU）：

```bash
python scripts/run_eiffel_fair_experiments.py \
  --suite all --generate-only --main-budget 6 --ablation-budget 3 \
  --gpu 1 --collision --output-dir results/eiffel_phase4_plan
```

正式主实验：

```bash
PYTHONPATH="$PWD/.venv-myl12-realmesh:$PWD/.venv-myl12-deps:$PWD/.venv-myl12-source/Depth-Anything-3-3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src" \
HF_HOME="$PWD/.venv-myl12-hf" HF_HUB_OFFLINE=1 \
.venv-myl12/bin/python scripts/run_eiffel_fair_experiments.py \
  --suite main --main-budget 6 --ablation-budget 3 --gpu 1 --collision \
  --output-dir results/eiffel_phase4_main_validated
```

本次运行环境：Linux 5.15、Python 3.10.4、PyTorch 2.4.1+cu121、PyTorch3D 0.7.9、
CUDA 12.1、driver 535.230.02、RTX 3090 24 GiB。DA3 model revision
`8615eefb62f2db4f8d6ebaa59160086981672829`，source revision
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`；严格离线读取已缓存 checkpoint。

## 六视图主结果

| 规划器 / depth | normalized coverage | 最终点数 | 轨迹长度 (m) | 在线轨迹时延 (s) | provider (s) | peak alloc / reserved (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SCONE / GT | 0.201126 | 8,196 | 37.862 | 10.061 | 0.012 | 1,866.8 / 2,278 |
| SCONE / DA3 | 0.058120 | 28,901 | 31.552 | 37.091 | 24.834 | 11,057.8 / 12,590 |
| MAGICIAN / GT | 0.011369 | 402 | 170.379 | 8.671 | 0.011 | 1,497.5 / 1,722 |
| MAGICIAN / DA3 | 0.082847 | 35,016 | 157.758 | 42.084 | 31.829 | 11,059.7 / 12,114 |

完整 coverage 序列：

```text
SCONE/GT       [0, 0.006169, 0.013612, 0.023197, 0.119503, 0.201126]
SCONE/DA3      [0, 0,        0,        0,        0,        0.058120]
MAGICIAN/GT    [0, 0,        0,        0,        0,        0.011369]
MAGICIAN/DA3   [0, 0,        0.037472, 0.050269, 0.071121, 0.082847]
```

这些不是“GT 一定优于 DA3”或“MAGICIAN 一定优于 SCONE”的跨场景排名。GT 是深度
upper bound，但 policy 会因其在线几何状态选择不同轨迹；本实验比较的是完整闭环系统，
不是固定轨迹上的纯 depth replacement。

## Depth、geometry 与 confidence

离线诊断只在每条轨迹 renderer GT mask 与在线 planning mask 的交集上计算：

| 轨迹 | depth MAE / RMSE (m) | AbsRel | δ<1.25 | ray-geometry MAE / RMSE (m) | test-fit scale-only factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| SCONE / GT | 0.000008 / 0.000012 | ~0 | 1.000 | 0 / 0 | 1.000000 |
| SCONE / DA3 | 179.835 / 283.842 | 0.695 | 0.287 | 199.262 / 308.906 | 1.1796 |
| MAGICIAN / GT | 0.000001 / 0.000002 | ~0 | 1.000 | 0 / 0 | 1.000000 |
| MAGICIAN / DA3 | 9.635 / 13.536 | 0.406 | 0.484 | 11.530 / 15.965 | 0.8623 |

SCONE/DA3 的超大 aggregate error 主要来自它最后把相机抬到
`[-5,87.5,5]` 后的轨迹分布，不能解释为同一图像上 DA3 在 SCONE 中“模型更差”。

DA3 confidence 与绝对 depth error 的 Pearson 相关在 SCONE 六视图为 `-0.289`，
MAGICIAN 六视图为 `-0.186`；MAGICIAN 三视图基线则为 `+0.566`。符号随轨迹改变，
confidence quartile 也不呈稳定单调误差关系，因此未校准前继续保持
`da3_confidence_percentile=null` 是合理默认值。

## 三视图 smoke 诊断

统一 mapping gathering multiplier 为 `1.0` 后，三视图仍复现：

```text
SCONE/DA3     coverage [0, 0, 0]
MAGICIAN/DA3 coverage [0, 0, 0.037472]
```

直接原因是 policy 轨迹，而不是先前 MAGICIAN 的 2× 点云 gathering radius：

- SCONE 先在同一位置 `[-5,79.167,5]` 连续改变 elevation：`60 → 30 → 0`，三帧
  camera center 没有平移，DA3 pose conditioning 一直关闭；六视图到第 6 帧才移动并
  得到 `0.058120` coverage。
- MAGICIAN 沿 y 轴连续移动：`79.167 → 70.833 → 62.500`；第三帧满足非共线/秩门禁，
  pose conditioning 开启，coverage 达 `0.037472`。
- 2× gathering radius 使三视图点数从 17,508 翻倍到 35,019，但 coverage 仅从
  `0.037472` 变为 `0.037370`。因此 Stage 3 的 `0` vs `0.0378` 主要是短预算下的
  轨迹/pose-conditioning 差异，不是 mapping 点密度差异。

## 消融

所有实跑消融固定 MAGICIAN/DA3、Eiffel、起点 0、seed、3 次观测、collision=true。

| 变体 | coverage | 最终点数 | depth MAE (m) | δ<1.25 | provider (s) | peak alloc (MiB) | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline (window 3, res 504, height scale, mapping 1×) | 0.037472 | 17,508 | 13.528 | 0.462 | 约 21 | 11,060 | reference |
| confidence p50 | 0.037472 | 17,508 | 13.528 | 0.462 | 21.427 | 11,060 | confidence=1 ties，实际未过滤 |
| confidence p75 | 0.037472 | 17,508 | 13.528 | 0.462 | 20.327 | 11,060 | 前帧 p75 仍为 1，仍未过滤 |
| confidence p90 | 0.008922 | 7,002 | 14.826 | 0.395 | 20.375 | 11,059 | 后两帧只保留 10%，coverage 大幅下降 |
| width scale `0.23766064` | 0.036452 | 17,508 | 15.253 | 0.461 | 19.866 | 11,060 | 与 height scale coverage 差 0.00102 |
| window 1 | 0 | 17,508 | 21.737 | 0 | 33.779 | 10,751 | scale-fit=3.976，失效 |
| process res 336 | 0.032221 | 17,508 | 13.949 | 0.273 | 21.860 | 10,790 | 省约 270 MiB，coverage/δ 降低 |
| mapping radius 2× | 0.037370 | 35,019 | 13.528 | 0.462 | 20.845 | 11,060 | 点数翻倍，coverage 几乎不变 |
| carving tolerance 5 | 0.037472 | 17,508 | 13.528 | 0.462 | 20.679 | 11,060 | 三视图无可测变化 |

Eiffel 高度标定与 125 m ground-width 交叉检查相差约 10%。合理 scale sensitivity
区间用两个端点 `[0.23766064, 0.2641176363636364] scene units/m` 覆盖；实跑了两端，
保留 midpoint 入口但未运行。

资源/时间限制下未运行但入口完整保留：confidence p25、scale midpoint、window 5、
resolution 672、mapping radius 0.5×、carving tolerance 20。矩阵 manifest 会逐项列出
`unselected_ablations`，不会把未运行配置描述成结果。

## 运行证据与失败披露

- 主实验：`results/eiffel_phase4_main_validated/{manifest.json,executions.json}` 和每个
  run 目录下的 config、log、online/diagnostic JSON；4/4 exit 0。
- 六个主消融：`results/eiffel_phase4_ablations/`；6/6 exit 0。
- confidence p75/p90 tie 诊断：`results/eiffel_phase4_confidence_p75/` 与
  `results/eiffel_phase4_confidence_p90/`；均 exit 0。
- `python -m unittest discover -s tests -v`：40 tests，全部通过。
- `scripts/validate_pytorch3d_runtime.py --device cuda:1`：5/5 checks 通过。

在 accepted evidence 之前有两个不计入结论的失败尝试：一次在线 coverage tuple 的
telemetry 序列化错误；一次未设置 `HF_HOME` 导致 Hugging Face 尝试联网并超时。
两项都未产生完成轨迹。最终 DA3 运行使用 strict offline cache，且四条主结果来自同一
Torch 2.4.1/PyTorch3D 0.7.9 runtime。
