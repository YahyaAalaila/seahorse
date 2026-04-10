# HawkesNest Hard Benchmark Suite v2

Generated at: 2026-04-10T17:57:09.980245
Representative seed: 50

## pulse

- Failure mode: Targets separable or factorized space-time models by shifting the triggering mode with lag.
- Difficulty metric: pulse_separability_gap
- P0: wave_speed_v=0.0, metric=0.000000, mean=104.6, p95=125.0, max=131
- P1: wave_speed_v=0.15, metric=0.437114, mean=104.0, p95=124.0, max=131
- P2: wave_speed_v=0.3, metric=0.601958, mean=103.7, p95=124.0, max=129
- P3: wave_speed_v=0.6, metric=0.741778, mean=103.2, p95=123.0, max=127
- P4: wave_speed_v=0.9, metric=0.806100, mean=102.8, p95=122.0, max=127
- P5: wave_speed_v=1.2, metric=0.843728, mean=102.4, p95=122.0, max=127

## echo

- Failure mode: Targets models that effectively assume one temporal scale and cannot fit both burst and echo.
- Difficulty metric: echo_single_scale_gap
- E0: echo_ratio_rho=1.0, metric=0.022392, mean=102.2, p95=121.1, max=127
- E1: echo_ratio_rho=3.0, metric=0.978670, mean=102.8, p95=123.0, max=127
- E2: echo_ratio_rho=10.0, metric=1.081027, mean=104.8, p95=126.0, max=128
- E3: echo_ratio_rho=30.0, metric=1.096816, mean=110.0, p95=134.0, max=141
- E4: echo_ratio_rho=100.0, metric=1.101219, mean=118.8, p95=144.0, max=151
- E5: echo_ratio_rho=300.0, metric=1.102377, mean=124.5, p95=155.1, max=157

## regime

- Failure mode: Targets stationary-background models by forcing abrupt spatial regime shifts over time.
- Difficulty metric: regime_shift_js
- R0: regime_amplitude=0.0, metric=0.000000, mean=102.7, p95=116.0, max=131
- R1: regime_amplitude=3.0, metric=0.004115, mean=108.0, p95=124.0, max=127
- R2: regime_amplitude=6.0, metric=0.012512, mean=109.7, p95=126.0, max=128
- R3: regime_amplitude=9.0, metric=0.022712, mean=112.2, p95=133.1, max=136
- R4: regime_amplitude=12.0, metric=0.033753, mean=116.9, p95=136.1, max=145
- R5: regime_amplitude=16.0, metric=0.048988, mean=120.5, p95=140.1, max=152

## topology

- Failure mode: Targets Euclidean continuous-space models by replacing flat geometry with graph-geodesic support.
- Difficulty metric: topology_distortion
- T0: theta_topo=0.0, metric=0.000000, mean=105.6, p95=120.0, max=129
- T1: theta_topo=0.2, metric=0.541489, mean=103.6, p95=117.1, max=126
- T2: theta_topo=0.4, metric=0.612842, mean=103.7, p95=118.0, max=127
- T3: theta_topo=0.6, metric=0.700924, mean=103.8, p95=117.1, max=127
- T4: theta_topo=0.8, metric=0.824163, mean=104.2, p95=119.0, max=127
- T5: theta_topo=0.99, metric=0.961805, mean=108.2, p95=121.1, max=138
