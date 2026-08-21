# PAN-14 paper Stage A final review

Review time: 2026-08-21 (UTC+8).  Scope: PAN-12 paper Stage A only.

## Decision

PASS.  The evidence closes the minimum credible UE5 observation-frontend
contribution.  Paper-stage UE5 development stops after this review.  Main
quantitative experiments, ablations, and long trajectories continue on the
original MAGICIAN/PyTorch3D pipeline.

PAN-12 remains an active two-stage roadmap because post-paper Stage B is still
pending.  This review does not authorize or start `PIONEER / UE5 Open Dataset
v1`, PAN-22, or PAN-24 through PAN-28.

## Ordered gate audit

| Gate | Status | Commit | Acceptance evidence |
| --- | --- | --- | --- |
| PAN-19 contract/validator | PASS / Done | `2a5afcb427821504ec970e56e0fba8b9f4172008` | `pan-19-contract/acceptance/20260821T123305Z`, checksum-list `264979705cf323119bc27c0b9025a469a98b43ca9b5d31d7506677e58e13b8fd` |
| PAN-23 UE5 skeleton/analytic rig | PASS / Done | `c0fb7d85c746acacaa2ed8bad354d97303960503` | `pan-23-ue5/acceptance/20260821T124300Z`, checksum-list `90fc746429c9908cd225b12d1af88f347d5bbb560a3784664c289349ed989644` |
| PAN-15 real RGB-D capture | PASS / Done | `058a0c92206d66752877a69addb2fc4c5779f566` | analytic checksum-list `782f5d77e59dbc06b18e548d27ec7df48d5efb5aac1515227d5e1b1af54b5d59`; HKUST `faf54b7e41d36be8644db72b56db2a434ad5f30b96ad5c4d386608051498f909` |
| PAN-20 geometry conformance | PASS / Done | `a4815c4c39b701222e0ec945e578e547ccdb8665` | `pan-20-conformance/acceptance/20260821T132821Z`, checksum-list `4e3f4159b9591ba8d9e4b3606a4cbb3c1659fcbe181f63f5ff7507466ade7146` |
| PAN-21 two-observation loop | PASS / Done | `b063d812b5fc9c1e2003ea32e08a507e4ad37f21` | `pan-21-smoke/acceptance/20260821T140700Z`, checksum-list `a9e70a9151037d7631b659a75676d792bb185716f89b411799db2100651334df` |

All five task worktrees were re-read during final review.  Their branches point
to the commits above and have no tracked or untracked changes.

## PAN-14 Definition of Done audit

1. **Analytic depth and axes:** PASS.  PAN-19 verifies camera-z to range at
   centre/edge/corner.  PAN-20 verifies actual UE RGBA16F camera-z units,
   pixel-centre intrinsics, all six marker directions, rotations, masks, and
   point-cloud geometry.  Analytic plane maximum range error is 0.00432 m.
2. **Replayable real HKUST fixture:** PASS.  PAN-15 publishes one atomic,
   checksummed six-face RGB-D fixture (108,506 valid depths and 284,710 explicit
   no-hits); PAN-21 adds two real execution-position fixtures p0 and p1.
3. **Existing PAN-10 path:** PASS.  PAN-21 changes representation only, then
   calls existing `process_cubemap_observation` and `update_proxy_state`.
   PAN-10 six-face fusion, visibility union, gain deduplication, and the main
   Planner algorithm were not rewritten.
4. **Shortest closed loop:** PASS.  Current PIONEER Planner consumes real UE5
   p0 and selects `[5,3,2]` from `[5,3,1]`; UE5 captures that p1; both real
   bundles update the same map path.  Acceptance records exactly two
   observations and one legal move.
5. **Atomic failure:** PASS.  Removing the `down` face is rejected before the
   frame counter or surface/covered/proxy hashes change; the next legal p0 is
   accepted.
6. **Sky, sun, shadow qualitative evidence:** PASS.  PAN-23's analytic level
   contains SkyAtmosphere, DirectionalLight, SkyLight, cast shadows, and open
   sky; its six RGB faces and PAN-21's combined cubemap/mask/point-cloud/move
   figure are checksummed.  VolumetricCloud is correctly treated as optional.
7. **Open-source/reproduction surface:** PASS.  The UE5 project skeleton,
   analytic level and rig, Python contract/adapter/validator, capture and smoke
   commands, synthetic and real fixtures, logs, hashes, and provider-switch
   documentation are committed across the ordered task branches.
8. **Paper claim boundary:** PASS.  UE5 demonstrates frontend compatibility and
   migration feasibility only.  It is not used for future Beam Search candidate
   renders or claimed as a main quantitative pipeline.

## Final-review commands actually run

```bash
git -C /home/ubuntu/Projects/Pioneer/repository worktree list --porcelain
git -C <each PAN-19/PAN-23/PAN-15/PAN-20/PAN-21 worktree> \
  branch --show-current
git -C <each worktree> rev-parse HEAD
git -C <each worktree> status --porcelain=v1 --untracked-files=all

sha256sum <PAN-19/PAN-23/PAN-15-analytic/PAN-15-HKUST/PAN-20/PAN-21 checksum-list>
(cd <each acceptance root> && sha256sum -c <checksum-list>)
```

Every checksum verification returned `OK`.  PAN-21 head also passed:

```bash
/home/ubuntu/anaconda3/envs/magician_mve/bin/python \
  -m unittest discover -s tests -v
```

Result: 174 tests, `OK`, exit 0.  Log:
`/home/ubuntu/Projects/Pioneer/experiments/pan-21-smoke/tests/20260821T141000Z/unittest.log`,
SHA-256 `f6fe2be2ecc786fb58d17218def63194ec4a17843667a1712897325ee5a73716`.

## Known limitations retained as paper boundaries

- UE depth is RGBA16F camera-z with binary16 quantization and a 65,504 UE-unit
  overflow/no-hit ceiling.
- HKUST geometry evidence is fixed-pose/coarse rather than exhaustive
  multi-scene conformance; the largest reported far-boundary seam p95 is 9.40 m
  (3.44% relative), while the robust median/relative gate passes.
- PAN-21 is debug-only and coverage-non-comparable.  The exchange is offline
  file/manifest, with no RPC, throughput, long-trajectory, main-table,
  ablation, or broad fault-injection claim.
- The acceptance audit map hashes the exact PAN-10 fused payload; the separate
  authoritative Planner run exercises the full MAGICIAN scene objects for real
  p0 before selecting p1.
- No push was requested or performed.  Integration/publication is outside this
  review.

## Stop record

At review time PAN-22 and PAN-24 through PAN-28 are all Backlog.  No Stage B
renderer, weather matrix, ERP export, data split, baseline, or release work was
started.  Further paper-stage UE5 development is out of scope after this PASS.
