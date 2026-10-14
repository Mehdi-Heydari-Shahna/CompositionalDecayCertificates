#!/bin/bash
# Regenerates the results of the paper from the code of this repository and writes them to results/.
#
#   bash reproduce.sh checks       pointwise checks and runs on the compiled models (Sections 6.3, 7.1, 7.2, 7.5, 7.8)
#   bash reproduce.sh analysis     certificates, refresh, synthesis and decay runs (Sections 7.1-7.5, 7.8)
#   bash reproduce.sh segments     carrying segment at plant steps of 2 to 0.25 ms on both plants (Section 7.6, Table 3)
#   bash reproduce.sh loading      MuJoCo missions with the grasp loads of every plant step (Sections 7.6, 7.8)
#   bash reproduce.sh robustness   carry stage under sampling, model error, noise and delay (Section 7.7)
#   bash reproduce.sh figures      Figs. 3 and 5-12 from the results
#   bash reproduce.sh tests        test suite of the framework package code/Team_CMG_PACDM
#   bash reproduce.sh              all of the above, in this order
#
# The 40 missions and the decay tests of the mission controller (Sections 7.5 and 7.6) are runs of the workflows of
# code/Team_CMG_PACDM; README.md ("Missions and decay tests") gives their commands.  Their reports are stored in
# code/Team_CMG_PACDM/reference_results/fitted_feedforward/, and results/checks/decay.* is recomputed here from the
# stored decay runs.
#
# Environment: PYTHON (default: python).  The console output of every step goes to results/logs/<step>.log.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PYTHON:-python}
C="$HERE/code/checks"
A="$HERE/code/analysis"
F="$HERE/code/figures"
LOGS="$HERE/results/logs"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run() {             # run <log name> <script> [arguments...]
    local name=$1
    shift
    mkdir -p "$LOGS"
    echo "== $(basename "$1") ${*:2}"
    "$PY" "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED: results/logs/$name.log"; tail -20 "$LOGS/$name.log"; exit 1; }
}

checks() {
    run survey "$C/survey.py"                                            # carrying configurations, dimensions (Fig. 4)
    run exp1_identity "$C/exp1_identity.py"                              # Proposition 3 at random states
    run exp2_modular "$C/exp2_modular.py"                                # scalar shares for gains designed robot by robot
    cp "$HERE/results/checks/exp2_modular.json" "$HERE/results/analysis/exp2_share_gains.json"      # Fig. 6(a)
    run exp3_events "$C/exp3_events.py"                                  # Lemma 21, Proposition 22, six teams
    run exp6_redundant "$C/exp6_redundant.py"                            # Lemma 2 with 28 rows of rank 26
    run exp7_actuator_reference "$C/exp7_actuator_reference.py"          # Theorem 19(a) at random states
    run exp5_friction_pointwise "$C/exp5_friction_pointwise.py"          # Proposition 23 on the Husky teams
    run exp5_husky2_floor025 "$C/exp5_friction_sim.py" husky2 --mu 0.25 --tag floor025
    run exp5_husky2_scrub "$C/exp5_friction_sim.py" husky2 --mu 0.25 --scrub 0.04 --tag scrub
    run exp5_husky2_floor080 "$C/exp5_friction_sim.py" husky2 --tag floor080
    run sim_team3_events "$C/sim.py" team3 --T 3 --amp-scale 0.5 --events                        # Fig. 3
    run sim_team3_events_jump "$C/sim.py" team3 --T 3 --amp-scale 0.5 --events --release-jump 0.15 --tag events_jump
    run decay "$C/decay.py" "$HERE/code/Team_CMG_PACDM/reference_results/fitted_feedforward/decay"  # Fig. 8
}

analysis() {
    run snapshots "$A/snapshots.py"                                      # the 24 navigation snapshots per team
    run check_table2 "$A/check_table2.py"                                # Section 7.2, Fig. 6(b)
    run check_realizability "$A/check_realizability.py"
    run check_refresh "$A/check_refresh.py"
    run verify_endpoint "$A/verify_endpoint.py"                          # Section 7.3
    run verify_new_theorems "$A/verify_new_theorems.py"
    run verify_edge_arity "$A/verify_edge_arity.py"
    run run_synth_tables "$A/run_synth_tables.py"                        # Table 2
    run run_admm_team2 "$A/run_admm_team.py" team2 200
    run sweep_admm "$A/sweep_admm.py"
    run check_gain_limits "$A/check_gain_limits.py"
    for t in team2 team3 ridgeback2 ridgeback3; do                       # Fig. 7
        run "decay_gains_$t" "$A/run_decay_gains.py" "$t"
        run "decay_gains_${t}_1ms" "$A/run_decay_gains.py" "$t" 0.001 0.001 _1ms
    done
    run check_decay_convergence "$A/check_decay_convergence.py"
    run check_decay_renewal10ms "$A/check_decay_renewal10ms.py"
    run check_friction_slow_mode "$A/check_friction_slow_mode.py"        # Section 7.8
}

segments() {
    for t in team2 team3 ridgeback2 ridgeback3; do
        run "segment_$t" "$A/run_segment_compare.py" "$t" 0.002 0.001 0.0005 0.00025
        run "segment_mujoco_$t" "$A/run_segment_compare.py" mujoco "$t" 0.002 0.001 0.0005 0.00025
    done
    run summarize_segments "$A/summarize_segments.py"
}

loading() {
    for t in team2 team3 ridgeback2 ridgeback3; do
        run "native_loading_$t" "$A/run_native_loading.py" "$t" fitted05
    done
    run native_loading_husky2 "$A/run_native_loading.py" husky2 desired
    run native_loading_husky3 "$A/run_native_loading.py" husky3 desired
    run summarize_native_loading "$A/summarize_native_loading.py"
    run check_native_inference "$A/check_native_inference.py"
}

robustness() {
    # the script continues an existing results file, so the recorded one is replaced
    rm -f "$HERE/results/checks/exp12_ridgeback3.json" "$HERE/results/checks/exp12_team2.json"
    run exp12_ridgeback3 "$C/exp12_robustness.py" ridgeback3 --only sampling,mass,noise,delay,half
    run exp12_team2 "$C/exp12_robustness.py" team2 --only sampling,mass,noise,delay,half --masses 0.7,1.3
}

figures() {
    for f in "$F"/fig*.py; do
        run "$(basename "$f" .py)" "$f"
    done
}

tests() {
    (cd "$HERE/code/Team_CMG_PACDM" && "$PY" -m pytest -p no:cacheprovider && "$PY" -m pytest -p no:cacheprovider -m slow)
}

case "${1:-all}" in
    checks) checks ;;
    analysis) analysis ;;
    segments) segments ;;
    loading) loading ;;
    robustness) robustness ;;
    figures) figures ;;
    tests) tests ;;
    all) checks; analysis; segments; loading; robustness; figures; tests ;;
    *) echo "usage: bash reproduce.sh [checks|analysis|segments|loading|robustness|figures|tests|all]"; exit 2 ;;
esac
echo "done: results in $HERE/results"
