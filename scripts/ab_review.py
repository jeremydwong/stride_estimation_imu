"""A/B review of a scoring change: flip-through PDFs of every bout it changes.

Compares two versions of the pipeline over every matched bout of a session:
BEFORE and AFTER can differ in code (a source folder, or a git revision
checked out into a temporary worktree) and/or in module switches (e.g.
WALKBACK_CUT=0). Each version runs in its own Python subprocess, so two
copies of brock_functions never mix. Changed bouts are sorted with a
reviewer's labels, and each group becomes ONE PDF in which every affected
trial-rep appears as a BEFORE page followed by an AFTER page (flip with the
arrow keys), after an index page. changes.csv has the numbers.

What counts as "changed" (--compare):
  window  gait start or end moved > 0.05 s, or the step count changed
  speed   any of the above, or any step speed moved > 0.05 m/s

Groups (from the labels CSV; --focus = the categories the change targets):
  fixed          labelled with a --focus category, and changed
  still_missed   labelled with a --focus category, but NOT changed
  touched_clean  in a trial-rep the reviewer marked "no changes", and changed
  touched_other  any other changed bout
  requested      the trial-reps named with --trials (always rendered)

Examples
  # a switch, measured in isolation (same code)
  uv run --with pypdf python scripts/ab_review.py s07_s08 A_walkback --focus WALKBACK \\
      --before WALKBACK_CUT=0 CHAIN_BACK=0 --after WALKBACK_CUT=1 CHAIN_BACK=0
  # a code change: the committed version vs the working tree
  uv run --with pypdf python scripts/ab_review.py s07_s08 step_speed --focus FIRST LAST \\
      --before-rev HEAD --compare speed --trials 14.1 12.1 45.1
Output: <imu data>/figures/review/<date>_<name>/
"""
import argparse
import contextlib
import datetime
import io
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import warnings

warnings.simplefilter('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/Users/jeremy/Dropbox/Treadmill Brock 2025/imu data'
SESSIONS = {'s07_s08': 'imuData_s07_s08_20260708.h5'}
GROUPS = ('fixed', 'still_missed', 'touched_clean', 'touched_other', 'requested')


# ---------------------------------------------------------------- worker side
def _worker(args):
    """Runs inside a subprocess with ONE version of the code on sys.path."""
    sys.path.insert(0, args.src)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt                                # noqa: F401
    import brock_functions as bf
    import stride_imu as imu
    for k, v in parse_switches(args.switches).items():
        setattr(bf, k, v)
    feet = bf.load_available_feet(os.path.join(DATA, SESSIONS[args.session]))
    period = next(iter(feet.values())).period
    upgrade = getattr(bf, '_upgrade_columns', lambda d: d)
    aligned = upgrade(pd.read_csv(args.aligned))
    if args.mode == 'metrics':
        pairs = bf.feet_pairs_from_labels(feet)
        out = {}
        for r in aligned[aligned['start_s'].notna()].itertuples():
            subj = {'a': 's1', 'b': 's2'}[str(r.walker)[-1]]
            kw = {'expected_m': float(r.distance_m)}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = bf.bout_sync_strides_steps(pairs, subj, int(r.start_s / period),
                                                     int(r.stop_s / period), period, **kw)
            except TypeError:                      # older code: no expected_m
                with contextlib.redirect_stdout(io.StringIO()):
                    res = bf.bout_sync_strides_steps(pairs, subj, int(r.start_s / period),
                                                     int(r.stop_s / period), period)
            out[(int(r.trial), int(r.rep), int(r.bout))] = dict(
                start=res['t0_abs'] * period, end=res['end_abs'] * period,
                steps=len(res['steps']['time']),
                speed=np.round(np.asarray(res['steps']['frwd_speed'], float), 3),
                walked=float(imu.overhead_travel(res)['mean'])
                if hasattr(imu, 'overhead_travel') else float('nan'))
    else:              # render: one single-page PDF per trial-rep (figures with
        import matplotlib.pyplot as plt          # lambda axes don't pickle)
        job = pickle.load(open(args.reps, 'rb'))
        tag, color = job['tag'], job['color']
        out = {}
        for (t, r), lines in job['pages'].items():
            with contextlib.redirect_stdout(io.StringIO()):
                fig = bf.inspect_snipped_trial(feet, aligned, t, rep=r, session_tag='')
            fig.suptitle(f'{tag}  ·  trial {t} rep {r}', color=color, fontsize=15,
                         fontweight='bold', y=0.998, va='top')
            fig.text(0.5, 0.985, lines, ha='center', va='top', fontsize=8,
                     family='monospace', color='0.25', wrap=True)
            path = os.path.join(job['dir'], f'{tag}_{t}_{r}.pdf')
            fig.savefig(path)
            plt.close(fig)
            out[(t, r)] = path
    pickle.dump(out, open(args.out, 'wb'))


# ------------------------------------------------------------------ main side
def parse_switches(items):
    out = {}
    for it in items or []:
        k, v = it.split('=')
        out[k] = float(v) if '.' in v else (bool(int(v)) if v in ('0', '1') else int(v))
    return out


def resolve_src(src, rev, tmp, tag):
    """A source folder for one side: --*-src, a git revision, or the repo."""
    if src:
        return os.path.abspath(src)
    if rev:
        wt = os.path.join(tmp, f'wt_{tag}')
        subprocess.run(['git', '-C', REPO, 'worktree', 'add', '--detach', '-q', wt, rev],
                       check=True)
        return os.path.join(wt, 'src')
    return os.path.join(REPO, 'src')


def merge_pdfs(paths, out):
    """Concatenate PDFs: pypdf if available (uv run --with pypdf ...), else
    poppler's pdfunite."""
    try:
        from pypdf import PdfWriter
        w = PdfWriter()
        for p in paths:
            w.append(p)
        with open(out, 'wb') as fh:
            w.write(fh)
    except ImportError:
        if not shutil.which('pdfunite'):
            raise SystemExit('need pypdf (uv run --with pypdf ...) or pdfunite (poppler)')
        subprocess.run(['pdfunite', *paths, out], check=True)


def spawn(mode, src, switches, a, out, reps=None):
    cmd = [sys.executable, os.path.abspath(__file__), a.session, a.name, '--focus', 'x',
           '--_worker', mode, '--src', src, '--aligned', a.aligned, '--out', out]
    if switches:
        cmd += ['--switches', *switches]
    if reps:
        cmd += ['--reps', reps]
    subprocess.run(cmd, check=True, env={**os.environ, 'PYTHONPATH': src})
    return pickle.load(open(out, 'rb'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 1)[1])
    ap.add_argument('session', choices=sorted(SESSIONS))
    ap.add_argument('name', help='short change name, e.g. A_walkback')
    ap.add_argument('--focus', nargs='+', required=True,
                    help='label categories this change targets')
    ap.add_argument('--before', nargs='*', default=[], help='SWITCH=value ...')
    ap.add_argument('--after', nargs='*', default=[], help='SWITCH=value ...')
    ap.add_argument('--before-src'); ap.add_argument('--after-src')
    ap.add_argument('--before-rev'); ap.add_argument('--after-rev')
    ap.add_argument('--compare', choices=('window', 'speed'), default='window')
    ap.add_argument('--trials', nargs='*', default=[], help='always render, e.g. 14.1 12.1')
    ap.add_argument('--aligned', default=None, help='aligned CSV (default: cached)')
    ap.add_argument('--labels', default=None); ap.add_argument('--notes', default=None)
    # worker-only
    ap.add_argument('--_worker', choices=('metrics', 'render'), help=argparse.SUPPRESS)
    ap.add_argument('--src', help=argparse.SUPPRESS)
    ap.add_argument('--switches', nargs='*', default=[], help=argparse.SUPPRESS)
    ap.add_argument('--out', help=argparse.SUPPRESS)
    ap.add_argument('--reps', help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.aligned = a.aligned or os.path.join(DATA, 'cached data',
                                          f'brock_{a.session}_auto_aligned.csv')
    if a._worker:
        a.mode = a._worker
        return _worker(a)

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    labels = pd.read_csv(a.labels or os.path.join(REPO, 'reports', f'{a.session}_hallee_labels.csv'))
    notes = pd.read_csv(a.notes or os.path.join(REPO, 'reports', f'{a.session}_hallee_notes.csv'))
    lab = {(r.trial, r.rep, r.bout): (r.category, r.evidence) for r in labels.itertuples()}
    clean = {(r.trial, r.rep) for r in notes.itertuples() if r.no_change}
    wanted = {tuple(int(x) for x in s.split('.')) for s in a.trials}

    tmp = tempfile.mkdtemp(prefix='ab_review_')
    try:
        srcB = resolve_src(a.before_src, a.before_rev, tmp, 'before')
        srcA = resolve_src(a.after_src, a.after_rev, tmp, 'after')
        print(f'BEFORE code: {srcB}  switches {a.before or "-"}\n'
              f'AFTER  code: {srcA}  switches {a.after or "-"}')
        mB = spawn('metrics', srcB, a.before, a, os.path.join(tmp, 'mB.pkl'))
        mA = spawn('metrics', srcA, a.after, a, os.path.join(tmp, 'mA.pkl'))

        rows = []
        for k in sorted(mB):
            b, f = mB[k], mA[k]
            moved = (abs(f['start'] - b['start']) > 0.05 or abs(f['end'] - b['end']) > 0.05
                     or f['steps'] != b['steps'])
            if a.compare == 'speed' and not moved:
                moved = (len(f['speed']) != len(b['speed'])
                         or bool(np.any(np.abs(f['speed'] - b['speed']) > 0.05)))
            cat = lab.get(k, ('', ''))[0]
            if cat in a.focus:
                group = 'fixed' if moved else 'still_missed'
            elif (k[0], k[1]) in wanted:
                group = 'requested'
            elif not moved:
                continue
            elif (k[0], k[1]) in clean:
                group = 'touched_clean'
            else:
                group = 'touched_other'
            fmt = lambda v: ' '.join(f'{x:.2f}' for x in v[:4])
            rows.append(dict(group=group, trial=k[0], rep=k[1], bout=k[2],
                             label=cat or ('clean' if (k[0], k[1]) in clean else 'not reviewed'),
                             evidence=lab.get(k, ('', ''))[1],
                             steps_before=b['steps'], steps_after=f['steps'],
                             walked_before=round(b['walked'], 2), walked_after=round(f['walked'], 2),
                             start_shift_s=round(f['start'] - b['start'], 2),
                             end_shift_s=round(f['end'] - b['end'], 2),
                             gait_before_s=round(b['end'] - b['start'], 2),
                             gait_after_s=round(f['end'] - f['start'], 2),
                             speeds_before=fmt(b['speed']), speeds_after=fmt(f['speed'])))
        df = pd.DataFrame(rows)
        for t, r in wanted:                       # requested trial-reps always render
            if not len(df) or not ((df.trial == t) & (df.rep == r)).any():
                for k in [k for k in mB if k[:2] == (t, r)]:
                    b, f = mB[k], mA[k]
                    df = pd.concat([df, pd.DataFrame([dict(group='requested', trial=t, rep=r, bout=k[2],
                        label='requested', evidence='', steps_before=b['steps'], steps_after=f['steps'],
                        walked_before=round(b['walked'], 2), walked_after=round(f['walked'], 2),
                        start_shift_s=round(f['start'] - b['start'], 2),
                        end_shift_s=round(f['end'] - b['end'], 2),
                        gait_before_s=round(b['end'] - b['start'], 2),
                        gait_after_s=round(f['end'] - f['start'], 2),
                        speeds_before=' '.join(f'{x:.2f}' for x in b['speed'][:4]),
                        speeds_after=' '.join(f'{x:.2f}' for x in f['speed'][:4]))])])

        out = os.path.join(DATA, 'figures', 'review', f'{datetime.date.today():%Y-%m-%d}_{a.name}')
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, 'changes.csv'), index=False)
        pages = {}
        for (t, r), grp in (df.groupby(['trial', 'rep']) if len(df) else []):
            pages[(int(t), int(r))] = '   '.join(
                f"b{x.bout} [{x.label}] steps {x.steps_before}->{x.steps_after}, "
                f"gait {x.gait_before_s}->{x.gait_after_s} s"
                + (f", speeds {x.speeds_before} -> {x.speeds_after}"
                   if a.compare == 'speed' else '')
                for x in grp.itertuples())
        figs = {}
        for tag, color, src, sw in (('BEFORE', '0.35', srcB, a.before),
                                    ('AFTER', '#2f7a5f', srcA, a.after)):
            if pages:
                jp = os.path.join(tmp, f'job_{tag}.pkl')
                pickle.dump({'tag': tag, 'color': color, 'dir': tmp, 'pages': pages},
                            open(jp, 'wb'))
                figs[tag] = spawn('render', src, sw, a, os.path.join(tmp, f'f{tag}.pkl'), jp)
        desc = (f"BEFORE: {a.before_rev or a.before_src or 'working tree'} "
                f"{' '.join(a.before) or ''}\nAFTER:  {a.after_rev or a.after_src or 'working tree'} "
                f"{' '.join(a.after) or ''}\ncompare: {a.compare}")
        for g in GROUPS:
            sub = df[df['group'] == g] if len(df) else df
            if g == 'requested' and not len(sub):
                continue
            cover = os.path.join(tmp, f'cover_{g}.pdf')
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.05, 0.95, f'{a.session} · {a.name} · {g.replace("_", " ")}',
                     fontsize=18, fontweight='bold', va='top')
            fig.text(0.05, 0.90, desc, fontsize=9, family='monospace', va='top')
            cols = ['trial', 'rep', 'bout', 'label', 'steps_before', 'steps_after',
                    'walked_before', 'walked_after', 'gait_before_s', 'gait_after_s']
            if a.compare == 'speed':
                cols += ['speeds_before', 'speeds_after']
            body = 'none' if not len(sub) else sub[cols].to_string(index=False)
            fig.text(0.05, 0.80, 'Each trial-rep follows as a BEFORE page then an AFTER '
                     'page.\n\n' + body, fontsize=7.5, family='monospace', va='top')
            fig.savefig(cover)
            plt.close(fig)
            order = [cover]
            for t, r in (sorted({(int(t), int(r)) for t, r in zip(sub.trial, sub.rep)})
                         if len(sub) else []):
                order += [figs['BEFORE'][(t, r)], figs['AFTER'][(t, r)]]
            path = os.path.join(out, f'{a.name}_{g}.pdf')
            merge_pdfs(order, path)
            print(f'{g:13s} {len(sub):3d} bouts -> {path}')
        print('changes.csv ->', os.path.join(out, 'changes.csv'))
    finally:
        for tag in ('before', 'after'):
            wt = os.path.join(tmp, f'wt_{tag}')
            if os.path.isdir(wt):
                subprocess.run(['git', '-C', REPO, 'worktree', 'remove', '--force', wt])
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
