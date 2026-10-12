#!/home/arenalabs/miniconda3/bin/python3
"""STREAM C - HiPHI bulk staged downloader (download -> sha256 verify -> extract originals -> delete tar).

* Queue: logs/hiphi_part_priority.csv (rank by hours of HOI + non-foot ground-contact originals),
  minus parts already present at first start (foreign tar in raw/hiphi/data or extracted marker).
* Download: aria2c, 1 part at a time (link-capped), CONNS connections, resumable (.aria2 control file).
  The HF token is passed to aria2c on stdin (never written to disk / argv).
* Verify: size == archive_manifest.csv and sha256 == x-linked-etag from the HF resolve HEAD.
* Extract: identical to scripts/run_hiphi_part.sh:
    tar --use-compress-program=zstd -xf AR -C raw/hiphi/extracted/part_XXXX --exclude='*__mirror*' && touch .extracted
  then check motion-dir count against motion_to_part.csv, append part id to logs/hiphi_ready_parts.txt,
  delete the tar. No conversion is run here.
* Staging / watchdog: at most MAX_TARS *.tar.zst files in raw/hiphi/data (foreign ones included);
  a part is only started if  free(/home) - reserved - need >= MIN_FREE  and
  du(umr_factory_data) + reserved + need <= BUDGET; the running download is paused (aria2c SIGINT,
  resumable) whenever free(/home) < MIN_FREE, and resumed with hysteresis.
* Lifetime: hard deadline LIFETIME_H (default 36 h); also wrapped by `timeout` in the launcher.
Status: logs/hiphi_streamc_status.json (every 30 s), events: logs/hiphi_streamc_events.jsonl.
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import csv, fcntl, glob, hashlib, json, os, shutil, signal, subprocess, sys, threading, time

ROOT = f'{SRCFG.data_root}'
DATA = f'{ROOT}/raw/hiphi/data'
EXT = f'{ROOT}/raw/hiphi/extracted'
LOGS = f'{ROOT}/logs'
READY = f'{LOGS}/hiphi_ready_parts.txt'
STATUS = f'{LOGS}/hiphi_streamc_status.json'
EVENTS = f'{LOGS}/hiphi_streamc_events.jsonl'
STATE = f'{LOGS}/hiphi_streamc_state.json'
PRIO = f'{LOGS}/hiphi_part_priority.csv'
ARIA_LOG = f'{LOGS}/hiphi_streamc_aria2.log'
TOKEN_FILE = os.path.expanduser('~/.cache/huggingface/token')
URL = 'https://huggingface.co/datasets/noitomrobotics/HiPHI/resolve/main/data/HiPHI_data_part_{p}_of_0032.tar.zst'
ZSTD = shutil.which('zstd') or '/home/arenalabs/miniconda3/bin/zstd'
IFACE = os.environ.get('IFACE', 'wlx8c86dd7755c9')

GB = 1e9
MIN_FREE = float(os.environ.get('MIN_FREE_GB', 200)) * GB
RESUME_MARGIN = float(os.environ.get('RESUME_MARGIN_GB', 5)) * GB
BUDGET = float(os.environ.get('BUDGET_GB', 300)) * GB
MAX_TARS = int(os.environ.get('MAX_TARS', 3))
CONNS = int(os.environ.get('CONNS', 16))
DL_TIMEOUT = os.environ.get('DL_TIMEOUT', '5h')
LIFETIME = float(os.environ.get('LIFETIME_H', 36)) * 3600
EXTRACT_FRAC = 0.51          # measured on part 0029: 13.47 GB originals / 26.81 GB uncompressed = 0.502
T0 = time.time()

lock = threading.Lock()
stop_evt = threading.Event()


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def event(kind, **kw):
    rec = dict(t=now(), ts=round(time.time(), 1), ev=kind, **kw)
    with lock:
        with open(EVENTS, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)


def tar_path(p):
    return f'{DATA}/HiPHI_data_part_{p}_of_0032.tar.zst'


def part_dir(p):
    return f'{EXT}/part_{p}'


def free_bytes():
    return shutil.disk_usage('/home').free


def du_bytes(path=ROOT):
    try:
        out = subprocess.run(['du', '-s', '--block-size=1', path], capture_output=True, text=True, timeout=300).stdout
        return int(out.split()[0])
    except Exception:
        return -1


def alloc_bytes(path):
    try:
        return os.stat(path).st_blocks * 512
    except FileNotFoundError:
        return 0


def rx_bytes():
    try:
        return int(open(f'/sys/class/net/{IFACE}/statistics/rx_bytes').read())
    except Exception:
        return -1


def n_tars():
    return len(glob.glob(f'{DATA}/*.tar.zst'))


def net_ok():
    """True iff DNS + HTTPS to huggingface.co work right now (public metadata endpoint, no token)."""
    try:
        r = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', '--max-time', '20',
                            'https://huggingface.co/api/datasets/noitomrobotics/HiPHI'],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() in ('200', '401', '403')
    except Exception:
        return False


HARD_FAIL_MAX = int(os.environ.get('HARD_FAIL_MAX', 12))


def head_sha(p):
    tok = open(TOKEN_FILE).read().strip()
    for _ in range(3):
        try:
            r = subprocess.run(['curl', '-sS', '-I', '--max-time', '30', '-H', '@-', URL.format(p=p)],
                               input=f'Authorization: Bearer {tok}\n', capture_output=True, text=True, timeout=45)
            h = {}
            for line in r.stdout.splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    h[k.strip().lower()] = v.strip().strip('"')
            if 'x-linked-etag' in h:
                return h['x-linked-etag'], int(h.get('x-linked-size', -1))
        except Exception as e:  # noqa
            event('head_error', part=p, err=str(e)[:200])
        time.sleep(5)
    return None, -1


class Part:
    def __init__(self, row):
        self.p = row['part']
        self.rank = int(row['rank'])
        self.comp = int(row['compressed_bytes'])
        self.uncomp = int(row['uncompressed_bytes'])
        self.n_orig = int(row['n_orig'])
        self.prio_h = float(row['prio_h']); self.hoi_h = float(row['hoi_h']); self.gc_h = float(row['gc_h'])
        self.orig_h = float(row['orig_h'])
        self.est_extract = int(self.uncomp * EXTRACT_FRAC)
        self.state = 'queued'   # queued|downloading|paused|downloaded|verifying|extracting|ready|failed|foreign
        self.sha = None
        self.attempts = 0
        self.hard_fail = 0       # consecutive aria2 failures while the network was up
        self.net_fail = 0        # failures attributed to network outage (not counted toward give-up)
        self.retry_at = 0.0
        self.t_dl0 = self.t_dl1 = self.t_x0 = self.t_x1 = None
        self.dl_secs = 0.0
        self.extracted_bytes = None
        self.n_dirs = None
        self.err = None

    def reserve(self):
        """bytes this part will still add to /home before it reaches steady state (tar deleted)."""
        if self.state in ('ready', 'failed', 'foreign', 'queued'):
            return 0
        remaining_dl = max(0, self.comp - alloc_bytes(tar_path(self.p)))
        if self.state == 'extracting':
            done = du_bytes(part_dir(self.p))
            return max(0, self.est_extract - max(done, 0))
        return remaining_dl + self.est_extract

    def to_json(self):
        d = {k: v for k, v in self.__dict__.items()}
        d['bytes_on_disk'] = alloc_bytes(tar_path(self.p)) if self.state in ('downloading', 'paused', 'downloaded', 'verifying', 'extracting') else None
        return d


# ----------------------------------------------------------------------------------------------
class Streamc:
    def __init__(self):
        rows = list(csv.DictReader(open(PRIO)))
        self.parts = [Part(r) for r in rows]
        self.byid = {pt.p: pt for pt in self.parts}
        st = json.load(open(STATE)) if os.path.exists(STATE) else {}
        self.foreign = set(st.get('foreign', []))
        self.owned = set(st.get('owned', []))
        if not st:   # first start: anything already present is foreign (owned by the factory job)
            for pt in self.parts:
                if os.path.exists(tar_path(pt.p)) or os.path.exists(part_dir(pt.p) + '/.extracted'):
                    self.foreign.add(pt.p)
        ready = set(open(READY).read().split()) if os.path.exists(READY) else set()
        for pt in self.parts:
            if pt.p in self.foreign:
                pt.state = 'foreign'
            elif os.path.exists(part_dir(pt.p) + '/.extracted'):
                pt.state = 'ready'
                if pt.p not in ready:
                    self.append_ready(pt.p)
                if os.path.exists(tar_path(pt.p)) and not os.path.exists(tar_path(pt.p) + '.aria2'):
                    os.remove(tar_path(pt.p))
            elif os.path.exists(tar_path(pt.p)) and not os.path.exists(tar_path(pt.p) + '.aria2') \
                    and os.path.getsize(tar_path(pt.p)) == pt.comp and alloc_bytes(tar_path(pt.p)) >= pt.comp * 0.99:
                pt.state = 'downloaded'      # resumed run: verify + extract
            elif pt.p in self.owned and os.path.exists(tar_path(pt.p)):
                pt.state = 'paused'          # our own partial download from a previous run: resume it first
        self.save_state()
        self.net_up = True
        self.sess_b0 = None      # bytes-on-disk accounting for session average rate
        self.sess_bytes = 0
        self.sess_secs = 0.0
        self.cur = None          # Part being downloaded
        self.proc = None
        self.paused_reason = None
        self.rate_hist = []      # (t, cumulative downloaded bytes by us)
        self.rx_hist = []
        self.dl_cum = 0
        self.extract_q = [pt for pt in self.parts if pt.state == 'downloaded']
        self.extractor = threading.Thread(target=self.extract_loop, daemon=True)
        self.du_cache = (0, du_bytes())
        self.xproc = None

    def save_state(self):
        json.dump(dict(foreign=sorted(self.foreign), owned=sorted(self.owned)), open(STATE + '.tmp', 'w'))
        os.replace(STATE + '.tmp', STATE)

    def append_ready(self, p):
        with open(READY, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(p + '\n'); f.flush(); os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def du(self, max_age=120):
        t, v = self.du_cache
        if time.time() - t > max_age:
            self.du_cache = (time.time(), du_bytes())
        return self.du_cache[1]

    def reserved(self, exclude=None):
        return sum(pt.reserve() for pt in self.parts if pt is not exclude)

    def can_start(self, pt):
        others = n_tars() - (1 if os.path.exists(tar_path(pt.p)) else 0)
        if others >= MAX_TARS:
            return False, f'tar slots full ({others}/{MAX_TARS} other *.tar.zst in raw/hiphi/data)'
        need = pt.comp - alloc_bytes(tar_path(pt.p)) + pt.est_extract
        res = self.reserved(exclude=pt)
        fr = free_bytes()
        if fr - res - need < MIN_FREE:
            return False, f'disk: free {fr/GB:.1f} GB - reserved {res/GB:.1f} - need {need/GB:.1f} < {MIN_FREE/GB:.0f} GB'
        du = self.du()
        if du + res + need > BUDGET:
            return False, f'budget: du {du/GB:.1f} + reserved {res/GB:.1f} + need {need/GB:.1f} > {BUDGET/GB:.0f} GB'
        return True, ''

    # ---------------- download ----------------
    def start_download(self, pt):
        if pt.sha is None:
            pt.sha, size = head_sha(pt.p)
            if size not in (-1, pt.comp):
                event('size_mismatch_head', part=pt.p, head=size, manifest=pt.comp)
        tok = open(TOKEN_FILE).read().strip()
        inp = f'{URL.format(p=pt.p)}\n  header=Authorization: Bearer {tok}\n  out=HiPHI_data_part_{pt.p}_of_0032.tar.zst\n'
        cmd = ['timeout', '--signal=INT', '--kill-after=120', DL_TIMEOUT,
               'nice', '-n', '5', 'aria2c', '-i', '-', '-d', DATA,
               f'-x{CONNS}', f'-s{CONNS}', '-k', '8M', '-j', '1',
               '--file-allocation=none', '--continue=true', '--auto-file-renaming=false',
               '--allow-overwrite=false', '--max-tries=20', '--retry-wait=15', '--timeout=60',
               '--connect-timeout=30', '--auto-save-interval=30', '--summary-interval=60',
               '--console-log-level=warn', '--enable-color=false', '--show-console-readout=false',
               '--max-file-not-found=3', '--async-dns=false']
        self.owned.add(pt.p); self.save_state()
        logf = open(ARIA_LOG, 'a')
        logf.write(f'\n==== {now()} part {pt.p} attempt {pt.attempts + 1}\n'); logf.flush()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=logf, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
        self.proc.stdin.write(inp); self.proc.stdin.close()
        self.cur = pt
        pt.state = 'downloading'; pt.attempts += 1
        pt.t_dl0 = pt.t_dl0 or time.time()
        self.cur_t0 = time.time()
        self.cur_b0 = alloc_bytes(tar_path(pt.p))
        event('dl_start', part=pt.p, rank=pt.rank, attempt=pt.attempts, have_bytes=self.cur_b0,
              comp=pt.comp, sha=pt.sha, free_gb=round(free_bytes() / GB, 1))

    def stop_download(self, reason):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL); self.proc.wait()
        if self.cur:
            self.cur.dl_secs += time.time() - self.cur_t0
            self.sess_secs += time.time() - self.cur_t0
            self.sess_bytes += max(0, min(alloc_bytes(tar_path(self.cur.p)), self.cur.comp) - self.cur_b0)
            self.cur.state = 'paused'
            event('dl_pause', part=self.cur.p, reason=reason, have_bytes=alloc_bytes(tar_path(self.cur.p)))
        self.paused_reason = reason
        self.proc = None

    def poll_download(self):
        pt = self.cur
        rc = self.proc.poll()
        if rc is None:
            return
        dt = time.time() - self.cur_t0
        pt.dl_secs += dt
        f = tar_path(pt.p)
        got = max(0, min(alloc_bytes(f), pt.comp) - self.cur_b0)
        self.sess_secs += dt; self.sess_bytes += got
        ok = rc == 0 and os.path.exists(f) and not os.path.exists(f + '.aria2') and os.path.getsize(f) == pt.comp
        self.proc = None; self.cur = None
        if ok:
            pt.state = 'downloaded'; pt.t_dl1 = time.time(); pt.hard_fail = 0; pt.err = None
            event('dl_done', part=pt.p, secs=round(pt.dl_secs), attempt_secs=round(dt),
                  attempt_MBps=round(got / dt / 1e6, 2) if dt > 0 else None)
            with lock:
                self.extract_q.append(pt)
            return
        # failure: never lose the partial file; classify network outage vs. real failure
        up = net_ok()
        pt.err = f'aria2c rc={rc}' + ('' if up else ' (network down)')
        if not up:
            pt.net_fail += 1; pt.state = 'paused'; pt.retry_at = time.time() + 30
            self.net_up = False
            event('dl_fail_net', part=pt.p, rc=rc, attempt=pt.attempts, got_MB=round(got / 1e6, 1))
            return
        pt.hard_fail = pt.hard_fail + 1 if got < 50e6 else 1    # progress resets the streak
        backoff = min(600, 30 * 2 ** (pt.hard_fail - 1))
        event('dl_fail', part=pt.p, rc=rc, attempt=pt.attempts, hard_fail=pt.hard_fail,
              got_MB=round(got / 1e6, 1), backoff_s=backoff)
        if pt.hard_fail >= HARD_FAIL_MAX:
            pt.state = 'failed'
            for x in (f, f + '.aria2'):      # free the tar slot so the queue cannot deadlock
                if os.path.exists(x):
                    os.remove(x)
            event('dl_giveup', part=pt.p, hard_fail=pt.hard_fail)
        else:
            pt.state = 'paused'; pt.retry_at = time.time() + backoff

    # ---------------- extract ----------------
    def extract_loop(self):
        while not stop_evt.is_set():
            with lock:
                pt = self.extract_q.pop(0) if self.extract_q else None
            if pt is None:
                time.sleep(5); continue
            try:
                self.extract_one(pt)
            except Exception as e:
                pt.state = 'failed'; pt.err = f'extract exception {e!r}'[:300]
                event('extract_exception', part=pt.p, err=pt.err)

    def extract_one(self, pt):
        f = tar_path(pt.p)
        pt.state = 'verifying'
        t = time.time()
        if pt.sha is None:
            pt.sha, _ = head_sha(pt.p)
        h = hashlib.sha256()
        with open(f, 'rb') as fh:
            while True:
                b = fh.read(1 << 24)
                if not b:
                    break
                h.update(b)
                if stop_evt.is_set():
                    pt.state = 'downloaded'; return
        got = h.hexdigest()
        if pt.sha and got != pt.sha:
            event('sha_mismatch', part=pt.p, got=got, want=pt.sha)
            pt.sha_fail = getattr(pt, 'sha_fail', 0) + 1
            os.remove(f); pt.state = 'queued' if pt.sha_fail < 3 else 'failed'; pt.t_dl0 = None; pt.dl_secs = 0
            return
        event('sha_ok' if pt.sha else 'sha_unchecked', part=pt.p, sha=got, secs=round(time.time() - t, 1))
        # disk gate for extraction (normally already reserved before the download started)
        while not stop_evt.is_set():
            fr = free_bytes()
            if fr - pt.est_extract >= MIN_FREE and self.du(30) + pt.est_extract <= BUDGET + pt.comp:
                break
            if pt.state != 'wait_disk':
                event('extract_wait_disk', part=pt.p, free_gb=round(fr / GB, 1))
            pt.state = 'wait_disk'
            time.sleep(60)
        if stop_evt.is_set():
            pt.state = 'downloaded'; return
        pt.state = 'extracting'; pt.t_x0 = time.time()
        d = part_dir(pt.p)
        os.makedirs(d, exist_ok=True)
        cmd = ['timeout', '2h', 'nice', '-n', '10', 'ionice', '-c2', '-n7',
               'tar', f'--use-compress-program={ZSTD}', '-xf', f, '-C', d, '--exclude=*__mirror*']
        self.xproc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      start_new_session=True)
        out, _ = self.xproc.communicate()
        rc = self.xproc.returncode
        if rc != 0:
            pt.err = f'tar rc={rc}: {out[-300:]}'
            event('extract_fail', part=pt.p, rc=rc, out=out[-300:])
            pt.x_attempts = getattr(pt, 'x_attempts', 0) + 1
            if stop_evt.is_set():
                pt.state = 'downloaded'
            elif pt.x_attempts >= 3:
                # archive is bad (zstd XXH64 / tar error): drop it and re-download once more, never keep a dead tar slot
                pt.sha_fail = getattr(pt, 'sha_fail', 0) + 1
                shutil.rmtree(d, ignore_errors=True)
                if os.path.exists(f):
                    os.remove(f)
                pt.x_attempts = 0; pt.t_dl0 = None; pt.dl_secs = 0
                pt.state = 'queued' if pt.sha_fail < 3 else 'failed'
                event('extract_giveup_redownload' if pt.state == 'queued' else 'extract_giveup', part=pt.p)
            else:
                pt.state = 'downloaded'
                time.sleep(60)
                with lock:
                    self.extract_q.append(pt)
            return
        open(d + '/.extracted', 'w').close()
        pt.t_x1 = time.time()
        dirs = subprocess.run(['find', d, '-name', 'metadata.json'], capture_output=True, text=True).stdout.split()
        mirrors = sum('__mirror' in x for x in dirs)
        pt.n_dirs = len(dirs)
        pt.extracted_bytes = du_bytes(d)
        self.append_ready(pt.p)
        pt.state = 'ready'
        os.remove(f)
        self.du_cache = (0, 0)
        event('ready', part=pt.p, extract_secs=round(pt.t_x1 - pt.t_x0, 1), n_motion_dirs=pt.n_dirs,
              expected=pt.n_orig, mirror_dirs=mirrors, extracted_gb=round(pt.extracted_bytes / GB, 2),
              free_gb=round(free_bytes() / GB, 1))
        if pt.n_dirs != pt.n_orig or mirrors:
            event('count_mismatch', part=pt.p, n_motion_dirs=pt.n_dirs, expected=pt.n_orig, mirror_dirs=mirrors)

    # ---------------- status ----------------
    def write_status(self):
        t = time.time()
        if self.cur:
            cum = self.dl_cum + alloc_bytes(tar_path(self.cur.p))
        else:
            cum = self.dl_cum
        # monotone cumulative byte counter over our parts (ready parts count fully, others by bytes on disk)
        self.rate_hist.append((t, sum(pt.comp if pt.state == 'ready' else min(alloc_bytes(tar_path(pt.p)), pt.comp)
                                      for pt in self.parts if pt.p in self.owned)))
        self.rx_hist.append((t, rx_bytes()))
        self.rate_hist = [x for x in self.rate_hist if t - x[0] <= 900]
        self.rx_hist = [x for x in self.rx_hist if t - x[0] <= 900]

        def rate(h, win):
            pts = [x for x in h if t - x[0] <= win]
            if len(pts) < 2 or pts[-1][0] - pts[0][0] < 1:
                return None
            return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])
        r1, r10 = rate(self.rate_hist, 60), rate(self.rate_hist, 600)
        link = rate(self.rx_hist, 60)
        todo = [pt for pt in self.parts if pt.state not in ('ready', 'foreign', 'failed')]
        rem = sum(max(0, pt.comp - alloc_bytes(tar_path(pt.p))) if pt.state in ('downloading', 'paused', 'queued') else 0 for pt in todo)
        sb, ss = self.sess_bytes, self.sess_secs
        if self.cur and self.proc is not None:
            sb += max(0, min(alloc_bytes(tar_path(self.cur.p)), self.cur.comp) - self.cur_b0)
            ss += t - self.cur_t0
        sess = sb / ss if ss > 60 else None
        r_eta = r10 if (r10 and r10 > 1e4) else sess
        eta_s = rem / r_eta if r_eta and r_eta > 1e4 else None
        st = dict(
            t=now(), uptime_h=round((t - T0) / 3600, 3), deadline_in_h=round((LIFETIME - (t - T0)) / 3600, 2),
            current=self.cur.p if self.cur else None, paused_reason=self.paused_reason, net_up=self.net_up,
            our_rate_MBps_1m=round(r1 / 1e6, 3) if r1 is not None else None,
            our_rate_MBps_10m=round(r10 / 1e6, 3) if r10 is not None else None,
            session_avg_MBps=round(sess / 1e6, 3) if sess else None,
            session_downloaded_GB=round(sb / GB, 2), session_active_dl_h=round(ss / 3600, 2),
            link_rx_MBps_1m=round(link / 1e6, 3) if link is not None else None,
            remaining_download_GB=round(rem / GB, 2), eta_all_parts_h=round(eta_s / 3600, 2) if eta_s else None,
            free_home_GB=round(free_bytes() / GB, 1), du_factory_GB=round(self.du() / GB, 1),
            reserved_GB=round(self.reserved() / GB, 1), tars_on_disk=n_tars(),
            n_ready=sum(pt.state == 'ready' for pt in self.parts),
            n_todo=len(todo), extracted_total_GB=round(sum((pt.extracted_bytes or 0) for pt in self.parts) / GB, 2),
            ready_hoi_h=round(sum(pt.hoi_h for pt in self.parts if pt.state == 'ready'), 2),
            ready_orig_h=round(sum(pt.orig_h for pt in self.parts if pt.state == 'ready'), 2),
            limits=dict(MIN_FREE_GB=MIN_FREE / GB, BUDGET_GB=BUDGET / GB, MAX_TARS=MAX_TARS, CONNS=CONNS),
            parts=[dict(part=pt.p, rank=pt.rank, state=pt.state, comp_GB=round(pt.comp / GB, 2),
                        have_GB=round(min(alloc_bytes(tar_path(pt.p)), pt.comp) / GB, 2) if pt.state not in ('ready', 'foreign') else None,
                        dl_secs=round(pt.dl_secs), extracted_GB=round(pt.extracted_bytes / GB, 2) if pt.extracted_bytes else None,
                        n_dirs=pt.n_dirs, n_orig=pt.n_orig, hoi_h=pt.hoi_h, gc_h=pt.gc_h, err=pt.err,
                        attempts=pt.attempts, hard_fail=pt.hard_fail, net_fail=pt.net_fail)
                   for pt in self.parts])
        json.dump(st, open(STATUS + '.tmp', 'w'), indent=1)
        os.replace(STATUS + '.tmp', STATUS)

    # ---------------- main ----------------
    def run(self):
        event('start', pid=os.getpid(), foreign=sorted(self.foreign),
              queue=[pt.p for pt in self.parts if pt.state == 'queued'],
              free_gb=round(free_bytes() / GB, 1), du_gb=round(self.du() / GB, 1),
              limits=dict(MIN_FREE_GB=MIN_FREE / GB, BUDGET_GB=BUDGET / GB, MAX_TARS=MAX_TARS, CONNS=CONNS))
        self.extractor.start()
        last_status = 0; last_block = None
        while not stop_evt.is_set():
            if time.time() - T0 > LIFETIME:
                event('lifetime_reached'); break
            fr = free_bytes()
            if self.proc is not None:
                if fr < MIN_FREE:
                    self.stop_download(f'free {fr/GB:.1f} GB < {MIN_FREE/GB:.0f} GB')
                else:
                    self.poll_download()
            if self.proc is None:
                cand = self.cur if (self.cur and self.cur.state == 'paused') else None
                if cand is None:
                    cand = next((pt for pt in self.parts if pt.state in ('paused',)), None) or \
                           next((pt for pt in self.parts if pt.state == 'queued'), None)
                if cand is None and not any(pt.state in ('downloaded', 'verifying', 'wait_disk', 'extracting') for pt in self.parts) \
                        and not self.extract_q:
                    event('all_done'); break
                if cand is not None and not self.net_up and time.time() - getattr(self, 'last_net_chk', 0) > 30:
                    self.last_net_chk = time.time()
                    if net_ok():
                        self.net_up = True; event('net_up', part=cand.p)
                if cand is not None and not self.net_up:
                    if self.paused_reason != 'network down':
                        event('blocked', part=cand.p, why='network down')
                    self.paused_reason = 'network down'; last_block = 'network down'
                elif cand is not None and cand.retry_at > time.time():
                    self.paused_reason = f'backoff: retry {cand.p} in {cand.retry_at - time.time():.0f} s'
                elif cand is not None:
                    ok, why = self.can_start(cand)
                    if ok and cand.state == 'paused' and fr < MIN_FREE + RESUME_MARGIN:
                        ok, why = False, f'hysteresis: free {fr/GB:.1f} GB'
                    if ok:
                        self.paused_reason = None; last_block = None
                        self.start_download(cand)
                    else:
                        self.paused_reason = why
                        if why.split(':')[0] != (last_block or '').split(':')[0]:
                            event('blocked', part=cand.p, why=why)
                        last_block = why
            if time.time() - last_status > 30:
                try:
                    self.write_status()
                except Exception as e:
                    event('status_error', err=repr(e)[:200])
                last_status = time.time()
            time.sleep(10)
        self.shutdown()

    def shutdown(self):
        stop_evt.set()
        if self.proc is not None:
            self.stop_download('shutdown')
        xp = getattr(self, 'xproc', None)
        if xp is not None and xp.poll() is None:
            os.killpg(xp.pid, signal.SIGTERM)
        self.extractor.join(timeout=120)
        try:
            self.write_status()
        except Exception:
            pass
        event('exit')


def main():
    os.makedirs(LOGS, exist_ok=True)
    lf = open(f'{LOGS}/hiphi_streamc.lock', 'w')
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('another instance holds the lock; exiting'); sys.exit(1)
    lf.write(str(os.getpid())); lf.flush()
    s = Streamc()

    def on_sig(signum, frame):
        event('signal', sig=signum)
        stop_evt.set()
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    s.run()


if __name__ == '__main__':
    main()
