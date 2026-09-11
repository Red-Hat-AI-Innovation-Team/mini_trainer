#!/usr/bin/env python3
"""The benchmark's verdicts and the markdown that carries them, from the
results.json benchmark.sh wrote. Prints the markdown; exit 1 unless all
three configs completed (the other verdicts are information, not a
check). Rules, per the compile benchmark spec: steps 1-3 warm up; a
post-warmup step over 5x the config's median means a recompilation; the
PR's eager run must stay within 10% of main's; compiled must beat eager."""
import json
import statistics
import sys

WARMUP_STEPS = 3
RECOMPILE_FACTOR = 5.0
REGRESSION_TOLERANCE = 1.10
CONFIGS = ("baseline", "pr-eager", "pr-compiled")
MARKER = "<!-- minicloud-benchmark -->"


def _post_warmup(metrics):
    rows = []
    for i, m in enumerate(metrics, 1):
        if m.get("step", i) > WARMUP_STEPS and "time_per_batch" in m:
            rows.append(m)
    return rows


def _median_step(metrics):
    post = _post_warmup(metrics)
    return statistics.median(m["time_per_batch"] for m in post) if post else None


def verdicts(res):
    med = {n: _median_step(res.get(n, {}).get("metrics", [])) for n in CONFIGS}
    v = {"medians": med}
    v["completed"] = all(res.get(n, {}).get("rc") == 0 and med[n] is not None for n in CONFIGS)
    unstable = [n for n in CONFIGS if med[n]
                and any(m["time_per_batch"] > RECOMPILE_FACTOR * med[n]
                        for m in _post_warmup(res[n]["metrics"]))]
    v["recompiled_configs"] = unstable
    v["stable_steps"] = "pr-compiled" not in unstable     # eager outliers are a note, not a verdict
    v["no_regression"] = bool(med["baseline"] and med["pr-eager"]
                              and med["pr-eager"] <= REGRESSION_TOLERANCE * med["baseline"])
    v["compile_benefit"] = bool(med["pr-eager"] and med["pr-compiled"]
                                and med["pr-compiled"] < med["pr-eager"])
    return v


def _row(res, name, med):
    r = res.get(name, {})
    post = _post_warmup(r.get("metrics", []))
    tps = statistics.median(m["tokens_per_second"] for m in post if "tokens_per_second" in m) if post else None
    mem = max((m["peak_memory_usage_GB"] for m in r.get("metrics", []) if "peak_memory_usage_GB" in m), default=None)
    fmt = lambda x, p: f"{x:.{p}f}" if x is not None else "—"
    status = "✅" if r.get("rc") == 0 else f"❌ rc={r.get('rc')}"
    return f"| {name} | {status} | {fmt(med.get(name), 3)} | {fmt(tps, 0)} | {fmt(mem, 1)} |"


def render(res, v):
    ok = lambda b: "✅" if b else "❌"
    med = v["medians"]
    speed = (f" — compiled {med['pr-eager'] / med['pr-compiled']:.2f}× eager"
             if med.get("pr-eager") and med.get("pr-compiled") else "")
    lines = [MARKER,
             f"## MiniCloud benchmark — PR #{res['pr']} ({res['mode']}){speed}", "",
             f"baseline `{res['shas']['main']}` (main) vs PR head `{res['shas']['pr']}` — "
             f"{res['gpus']}×GPU on the isolated CI node, torch {res.get('torch', '?')}, {res['model']}, "
             f"{res['steps']} steps, synthetic 100×512, TESTING=true (SDPA)", "",
             "| config | run | median step (s) | tokens/s | peak mem GB |", "|---|---|---|---|---|"]
    lines += [_row(res, n, med) for n in CONFIGS]
    lines += ["",
              f"- {ok(v['completed'])} all three configs completed",
              f"- {ok(v['stable_steps'])} no recompilation in the compiled config"
              + (f" (⚠ step outliers >{RECOMPILE_FACTOR:g}× median in: {', '.join(v['recompiled_configs'])})"
                 if v["recompiled_configs"] else ""),
              f"- {ok(v['no_regression'])} PR-eager within {int((REGRESSION_TOLERANCE - 1) * 100)}% of baseline",
              f"- {ok(v['compile_benefit'])} compiled faster than eager"]
    failed = [n for n in CONFIGS if res.get(n, {}).get("rc") != 0]
    if failed:
        lines += ["", "Failed configs: " + ", ".join(f"`{n}`" for n in failed)
                  + " — the logs are on the job's record (`minic job files`)."]
    return "\n".join(lines) + "\n"


def main():
    res = json.load(open(sys.argv[1]))
    v = verdicts(res)
    sys.stdout.write(render(res, v))
    sys.exit(0 if v["completed"] else 1)


if __name__ == "__main__":
    main()
