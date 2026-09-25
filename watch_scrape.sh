#!/bin/bash
# Live scraper health: rate, outcomes, and failure modes. Ctrl-C to exit.
cd "$(dirname "$(readlink -f "$0")")"
while true; do
  L=$(ls -t logs/review_run_*.log | head -1)
  clear
  echo "=== $(date +%H:%M:%S)   log: $L"
  echo "shards: $(pgrep -c -f 'scrape_reviews_batch.py --concurrency')  browsers: $(pgrep -c -x camoufox-bin)"
  RATE=$(pgrep -af run_review_collection_guarded.py | grep -oP '(?<=--starts-per-minute )[0-9.]+' | head -1)
  echo "configured rate: ${RATE:-<not running>}/min"
  echo "TargetClosedError: $(grep -c TargetClosedError "$L")   version.json: $(grep -c version.json "$L")   challenge: $(grep -icE '/sorry/|GoogleChallengeError|automated-traffic challenge' "$L")"
  python - <<'PY'
import json,pathlib,collections,time,datetime
now=time.time(); b=collections.defaultdict(collections.Counter)
for p in pathlib.Path("data/reviews").glob("*.json"):
    m=p.stat().st_mtime
    if m<now-2400: continue
    try: d=json.loads(p.read_text())
    except Exception: continue
    b[datetime.datetime.fromtimestamp(m).strftime("%H:%M")[:-1]+"0"][d.get("outcome","?")]+=1
print(f"\n{'bucket':>8} {'n':>5} {'rate/min':>9} {'scraped':>8} {'error':>6} {'bot':>4}  err%")
for k in sorted(b):
    c=b[k]; n=sum(c.values())
    print(f"{k:>8} {n:5d} {n/10:9.1f} {c['scraped']:8d} {c['error']:6d} {c['bot_detection']:4d}  {c['error']/n:5.1%}")
PY
  echo
  grep -E "scraped=|Starting chunk|RECYCLE|PAUSED|FAILED" "$L" | tail -6
  sleep 30
done
