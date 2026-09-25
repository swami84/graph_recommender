#!/bin/bash
# Supervise the review collection:
#   - revert to 24/min on any explicit Google challenge
#   - restart the run if checkpoint writes stall (a wedged worker blocks the
#     whole pipeline behind asyncio.gather, which cost ~2h on 2026-09-03)
cd "$(dirname "$(readlink -f "$0")")"
RATE="${1:-36}"
STALL_SECS=${STALL_SECS:-900}
MAX_RESTARTS=${MAX_RESTARTS:-4}
DEADLINE=$(( $(date +%s) + 8*3600 ))
restarts=0
log(){ echo "$(date +%H:%M:%S)  $*"; }

newest_write_age() {   # seconds since the most recent checkpoint
  python - <<'PY'
import pathlib,time
mt=[p.stat().st_mtime for p in pathlib.Path("data/reviews").glob("*.json")]
print(int(time.time()-max(mt)) if mt else 999999)
PY
}

start_run() {   # $1 = rate, $2 = log suffix
  local NEW="logs/review_run_$(date +%Y%m%d_%H%M%S)_$2.log"
  nohup python run_review_collection_guarded.py --concurrency 12 --workers 4 \
    --starts-per-minute "$1" --start-jitter 0.4 --chunk-size 100 --max-reviews 200 \
    > "$NEW" 2>&1 &
  disown
  log "started run at $1/min -> $NEW"
}

stop_run() {
  for pat in run_review_collection_guarded.py run_review_collection.py "scrape_reviews_batch.py --concurrency"; do
    pgrep -f "$pat" | grep -v "^$$\$" | xargs -r kill 2>/dev/null
  done
  sleep 8
  pgrep -f "scrape_reviews_batch.py --concurrency" | xargs -r kill -9 2>/dev/null
  pkill -9 -x camoufox-bin 2>/dev/null
  sleep 3
}

challenged() {
  grep -qiE "/sorry/|GoogleChallengeError|automated-traffic challenge|challenge remains active|challenge returned during collection" "$1"
}

log "guard armed: rate=${RATE}/min, stall=${STALL_SECS}s, max_restarts=${MAX_RESTARTS}"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 60
  if ! pgrep -f run_review_collection_guarded.py >/dev/null; then
    log "collection finished or stopped; guard exiting"; exit 0
  fi
  L=$(ls -t logs/review_run_*.log | head -1)

  if challenged "$L"; then
    log "CHALLENGE DETECTED -- reverting to 24/min"
    grep -iE "challenge|/sorry/" "$L" | tail -3
    stop_run; RATE=24; start_run 24 revert24
    continue
  fi

  age=$(newest_write_age)
  if [ "$age" -gt "$STALL_SECS" ]; then
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      log "STALLED ${age}s and restart budget spent ($restarts); leaving down"; exit 11
    fi
    restarts=$((restarts+1))
    log "STALL: no checkpoint written for ${age}s -- restart $restarts/$MAX_RESTARTS at ${RATE}/min"
    stop_run; start_run "$RATE" "stallrestart${restarts}"
  fi
done
log "guard deadline reached; exiting"
