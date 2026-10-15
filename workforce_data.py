"""Validated monthly snapshots. No inferred staffing labels or exit causes."""
import numpy as np
import pandas as pd

EMPLOYEE_COLUMNS = ['month', 'employee_id', 'team', 'normal_hours', 'leave_hours',
                    'planned_leave_hours', 'ot_hours', 'capability']
ASSIGNMENT_COLUMNS = ['month', 'employee_id', 'project_id', 'share']
GATE_COLUMNS = ['month', 'project_id', 'gate_id', 'stage', 'difficulty',
                'original_due', 'planned_due', 'completed_at']
CONTEXT = ['project_count', 'difficulty', 'capability', 'multi_project_mean',
           'gate_due_next', 'gate_overdue', 'extension_days'] + [f'gate_{i}' for i in range(1, 6)]
RISK_FEATURES = CONTEXT + ['headcount', 'available_hours', 'ot_total', 'ot_mean',
                         'ot_ratio', 'leave_hours', 'joiners', 'leavers']


def _frame(frame, required, keys):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'필수 열 누락: {sorted(missing)}')
    out = frame[required].copy()
    if out.empty or out.duplicated(keys).any():
        raise ValueError(f'빈 자료 또는 중복 키: {keys}')
    for key in keys:
        if out[key].isna().any() or out[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'빈 키: {key}')
        out[key] = out[key].astype(str)
    if not out.month.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():
        raise ValueError('month는 YYYY-MM이어야 합니다.')
    return out


def validate_snapshots(employees, assignments, gates):
    e = _frame(employees, EMPLOYEE_COLUMNS, ['month', 'employee_id'])
    a = _frame(assignments, ASSIGNMENT_COLUMNS, ['month', 'employee_id', 'project_id'])
    g = _frame(gates, GATE_COLUMNS, ['month', 'project_id', 'gate_id'])
    if e.team.isna().any() or e.team.astype(str).str.strip().eq('').any():
        raise ValueError('팀이 필요합니다.')
    # Names are not ordinal encodings; an explicit mapping can be applied upstream.
    e['development_type'] = e.team.astype(str).str.extract(r'(?i)(cell|pack)', expand=False).str.title()
    if e.development_type.isna().any():
        raise ValueError('팀 이름에 Cell 또는 Pack을 명시하세요(조직 연결표에서 변환 가능).')
    for frame, columns in [(e, EMPLOYEE_COLUMNS[3:]), (a, ['share']), (g, ['stage', 'difficulty'])]:
        for c in columns:
            frame[c] = pd.to_numeric(frame[c], errors='coerce')
            if not np.isfinite(frame[c]).all() or (frame[c] < 0).any():
                raise ValueError(f'{c}: 유한한 0 이상 숫자가 필요합니다.')
    if (e.normal_hours <= 0).any() or (e.leave_hours > e.normal_hours).any() or (e.planned_leave_hours > e.normal_hours).any():
        raise ValueError('normal_hours는 양수이며 휴가를 포함한 소정시간; 휴가는 이를 초과할 수 없습니다.')
    if not e.capability.between(1, 5).all() or not g.difficulty.between(1, 5).all():
        raise ValueError('역량·난이도는 1~5입니다.')
    if not g.stage.isin(range(1, 6)).all():
        raise ValueError('Gate 단계는 1~5 정수입니다.')
    if not a.share.between(0, 1, inclusive='right').all():
        raise ValueError('share는 0 초과 1 이하 비율입니다(30%는 0.3).')
    sums = a.groupby(['month', 'employee_id']).share.sum()
    expected = pd.MultiIndex.from_frame(e[['month', 'employee_id']])
    if set(sums.index) != set(expected) or not np.allclose(sums, 1, atol=1e-6):
        raise ValueError('모든 직원·월에 배정이 필요하며 share 합은 1이어야 합니다.')
    for c in ['original_due', 'planned_due', 'completed_at']:
        raw = g[c].replace('', np.nan)
        dates = pd.to_datetime(raw, errors='coerce')
        if (raw.notna() & dates.isna()).any() or (c != 'completed_at' and dates.isna().any()):
            raise ValueError(f'{c}: 날짜 형식을 확인하세요.')
        g[c] = dates
    end = pd.to_datetime(g.month) + pd.offsets.MonthEnd(0)
    if (g.completed_at > end).any():
        raise ValueError('기준월 이후의 실제 완료일은 입력할 수 없습니다.')
    if (g.groupby(['project_id', 'gate_id']).original_due.nunique() > 1).any():
        raise ValueError('동일 Gate의 original_due는 변경할 수 없습니다.')
    for _, h in g.sort_values('month').groupby(['project_id', 'gate_id']):
        seen = h.completed_at.dropna()
        if seen.nunique() > 1 or (len(seen) and h.loc[h.month >= h.loc[seen.index[0], 'month'], 'completed_at'].isna().any()):
            raise ValueError('완료 기록이 이후 스냅샷에서 사라지거나 바뀌었습니다.')
    assigned = set(map(tuple, a[['month', 'project_id']].to_numpy()))
    covered = set(map(tuple, g[['month', 'project_id']].to_numpy()))
    if not assigned <= covered:
        raise ValueError('배정된 모든 프로젝트·월의 Gate 스냅샷이 필요합니다.')
    return e, a, g


def build_panels(employees, assignments, gates):
    """Return team-month panel and one forecast/label per unique project Gate.

    Gate target is missed frozen planned deadline, not causally attributed to a team.
    One Gate is evaluated once, the first time its deadline enters next month.
    A missing outcome snapshot is unknown, never a success.
    """
    e, a, g = validate_snapshots(employees, assignments, gates)
    e['available_hours'] = e.normal_hours - e.leave_hours
    e['allocation_count'] = e.set_index(['month', 'employee_id']).index.map(
        a.groupby(['month', 'employee_id']).size()).to_numpy()
    rows = a.merge(e, on=['month', 'employee_id'], validate='many_to_one')
    if (rows.groupby(['month', 'project_id']).development_type.nunique() > 1).any():
        raise ValueError('Cell/Pack 공동 프로젝트는 유형별 프로젝트 ID로 범위를 먼저 구분하세요.')
    project = rows.groupby(['month', 'project_id']).apply(lambda r: pd.Series({
        'development_type': r.development_type.iloc[0], 'headcount': r.share.sum(),
        'available_hours': (r.available_hours*r.share).sum(),
        'ot_total': (r.ot_hours*r.share).sum(),
        'ot_mean': np.average(r.ot_hours, weights=r.share),
        'capability': np.average(r.capability, weights=r.share),
    }), include_groups=False).reset_index()
    team_rows = []
    for (month, team), people in e.groupby(['month', 'team']):
        end = pd.Period(month, 'M').end_time.normalize()
        next_end = (pd.Period(month, 'M')+1).end_time.normalize()
        pa = rows[(rows.month == month) & (rows.team == team)]
        ids = pa.project_id.unique()
        gs = g[(g.month == month) & g.project_id.isin(ids)]
        active = gs[gs.completed_at.isna()]
        previous_month = str(pd.Period(month, 'M')-1)
        prev = e[(e.month == previous_month) & (e.team == team)]
        previous_known = previous_month in set(e.month)
        current_ids, old_ids = set(people.employee_id), set(prev.employee_id)
        # Known deadlines, including unfinished overdue Gates; do not silently ignore extensions.
        start = pd.Period(month, 'M').start_time
        reached = (gs.original_due <= end) | (gs.planned_due <= end)
        relevant = (gs.completed_at.isna() | (gs.completed_at >= start) |
                    gs.original_due.between(start, end) | gs.planned_due.between(start, end))
        due = gs[reached & relevant]
        missed = due.completed_at.isna() | (due.completed_at > due.original_due) | (due.completed_at > due.planned_due)
        record = dict(month=month, team=team, development_type=people.development_type.iloc[0],
            headcount=len(people), normal_hours=people.normal_hours.sum(),
            available_hours=people.available_hours.sum(), leave_hours=people.leave_hours.sum(),
            planned_leave_hours=people.planned_leave_hours.sum(), ot_total=people.ot_hours.sum(),
            ot_mean=people.ot_hours.mean(), ot_p90=people.ot_hours.quantile(.9),
            ot_ratio=people.ot_hours.sum()/people.normal_hours.sum(), capability=people.capability.mean(),
            project_count=len(ids), multi_project_mean=people.allocation_count.mean(),
            difficulty=gs.groupby('project_id').difficulty.mean().mean(),
            gate_due_next=int(((active.planned_due > end) & (active.planned_due <= next_end)).sum()),
            gate_overdue=int(((active.original_due <= end) | (active.planned_due <= end)).sum()),
            extension_days=float((gs.planned_due-gs.original_due).dt.days.clip(lower=0).mean()),
            gates_observed=len(due), gate_misses=int(missed.sum()),
            joiners=len(current_ids-old_ids) if previous_known else np.nan,
            leavers=len(old_ids-current_ids) if previous_known else np.nan)
        record.update({f'gate_{i}': int((active.stage == i).sum()) for i in range(1, 6)})
        team_rows.append(record)
    teams = pd.DataFrame(team_rows).sort_values(['month', 'team']).reset_index(drop=True)
    gate_rows, seen = [], set()
    for r in g.sort_values('month').itertuples():
        end = pd.Period(r.month, 'M').end_time.normalize()
        target = str(pd.Period(r.month, 'M')+1)
        target_end = pd.Period(target, 'M').end_time.normalize()
        key = (r.project_id, r.gate_id)
        if key in seen or pd.notna(r.completed_at) or not end < r.planned_due <= target_end:
            continue
        p = project[(project.month == r.month) & (project.project_id == r.project_id)]
        if p.empty:
            continue
        seen.add(key)
        later = g[(g.month == target) & (g.project_id == r.project_id) & (g.gate_id == r.gate_id)]
        actual = np.nan
        if len(later):
            completed = later.completed_at.iloc[0]
            actual = float(pd.isna(completed) or completed > r.planned_due)
        rec = p.iloc[0].to_dict()
        rec.update(source_month=r.month, target_month=target, entity=f'{r.project_id}/{r.gate_id}',
                   gate_id=r.gate_id, stage=r.stage, difficulty=r.difficulty,
                   days_to_due=(r.planned_due-end).days,
                   extension_days=max(0, (r.planned_due-r.original_due).days), actual=actual,
                   frozen_due=r.planned_due)
        gate_rows.append(rec)
    return teams, pd.DataFrame(gate_rows)


def generate_snapshots(months=18, teams=10, seed=42):
    """Synthetic demonstrations only; no assumed optimal staffing ground truth."""
    rng = np.random.default_rng(seed)
    periods = pd.period_range('2025-01', periods=months, freq='M')
    employees, assignments, gates = [], [], []
    for k, period in enumerate(periods):
        month = str(period)
        for j in range(teams):
            family = 'Cell' if j % 2 == 0 else 'Pack'
            team, project = f'{family}_{j:02d}', f'P{j:02d}'
            n = 8 + j % 4
            phase = (k + j) % 5 + 1
            for i in range(n):
                employee = f'{j:02d}-{i:03d}'
                ot = max(0., float(rng.normal(9 + 3*(phase == 5) + j % 3, 3)))
                employees.append([month, employee, team, 160 + 8*(k % 3), 8., 8., ot, 3.+j*.03])
                assignments.append([month, employee, project, 1.])
            # Snapshot current Gate + next Gate, with deterministic real-outcome variation.
            for due_k in [k, k+1]:
                due_period = periods[0] + due_k
                due = due_period.start_time + pd.Timedelta(days=19)
                completed = due + pd.Timedelta(days=8 if (due_k+j) % 4 == 0 else -1)
                visible = completed if completed <= period.end_time else pd.NaT
                gates.append([month, project, f'G{due_k:02d}', (due_k+j)%5+1,
                              3+j % 2, due, due, visible])
    return (pd.DataFrame(employees, columns=EMPLOYEE_COLUMNS),
            pd.DataFrame(assignments, columns=ASSIGNMENT_COLUMNS),
            pd.DataFrame(gates, columns=GATE_COLUMNS))
