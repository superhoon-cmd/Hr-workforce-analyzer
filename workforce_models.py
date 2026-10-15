"""Three distinct estimands; chronological, target-specific model selection."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import QuantileRegressor, Ridge, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from workforce_data import CONTEXT, RISK_FEATURES

ALGORITHMS = ['Baseline', 'Regression', 'Random Forest', 'XGBoost']
GATE_FEATURES = ['headcount', 'available_hours', 'ot_total', 'ot_mean', 'capability',
                 'stage', 'difficulty', 'days_to_due', 'extension_days']


@dataclass(frozen=True)
class Settings:
    stable_ot: float = 20.
    leave_tolerance: float = 0.
    min_peer_rows: int = 20
    min_peer_teams: int = 3
    min_history_months: int = 3
    min_train_rows: int = 20
    selection_months: int = 3
    min_validation_months: int = 2
    min_improvement: float = 0.

    def __post_init__(self):
        for key in ['stable_ot', 'leave_tolerance', 'min_improvement']:
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key}는 0 이상의 유한한 값이어야 합니다.')
        for key in ['min_peer_rows', 'min_peer_teams', 'min_history_months', 'min_train_rows',
                    'selection_months', 'min_validation_months']:
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key}는 양의 정수여야 합니다.')
        if self.min_validation_months > self.selection_months:
            raise ValueError('최소 평가월은 평가창보다 클 수 없습니다.')


def normalize_ot(panel, target_ot=10.):
    if not np.isfinite(target_ot) or target_ot < 0:
        raise ValueError('기준 OT는 0 이상의 유한한 값이어야 합니다.')
    out = panel[['month', 'team', 'headcount', 'available_hours', 'ot_total']].copy()
    planned_capacity = (panel.normal_hours-panel.planned_leave_hours)/panel.headcount + target_ot
    out['normalized_staffing'] = (panel.available_hours+panel.ot_total)/planned_capacity.where(planned_capacity > 0)
    out['target_ot'] = target_ot
    out['interpretation'] = '관측 시간투입 환산; 최소 정원 또는 미래 수요 예측 아님'
    return out


def stable_mask(panel, settings):
    return ((panel.ot_p90 <= settings.stable_ot) &
            (panel.leave_hours + settings.leave_tolerance >= panel.planned_leave_hours) &
            (panel.gates_observed > 0) & (panel.gate_misses == 0) &
            (panel.extension_days == 0))


def peer_staffing(panel, month, settings=Settings()):
    """Cell/Pack-specific regularized linear quantiles + local empirical peers.

    Only earlier known stable months are used. Neither HC nor OT, capacity,
    allocated FTE or their direct equivalents are regressors. P10/P50/P90 are
    conditional observed staffing quantiles, not confidence or causal intervals.
    """
    results = []
    for _, row in panel[panel.month == month].iterrows():
        history = panel[(panel.month < month) & (panel.development_type == row.development_type)]
        stable = history[stable_mask(history, settings)]
        rec = dict(month=month, team=row.team, development_type=row.development_type,
                   status='판단 유보', reason='', stable_rows=len(stable), peer_rows=0,
                   p10=np.nan, p50=np.nan, p90=np.nan, peer_p10=np.nan, peer_p50=np.nan,
                   peer_p90=np.nan, quantile_rearranged=False,
                   training_end=stable.month.max() if len(stable) else None)
        if (len(stable) < settings.min_peer_rows or stable.team.nunique() < settings.min_peer_teams
                or stable.month.nunique() < settings.min_history_months):
            rec['reason'] = '같은 유형의 안정운영 사례·팀·기간 부족'
        else:
            # Match meaningful domain attributes before standardized distance.
            peers = stable[(abs(stable.difficulty-row.difficulty) <= 1) &
                           (abs(stable.capability-row.capability) <= .75) &
                           (abs(stable.project_count-row.project_count) <= max(1, row.project_count*.5))]
            scale = stable[CONTEXT].std().replace(0, 1).fillna(1)
            distance = np.sqrt((((peers[CONTEXT]-row[CONTEXT].astype(float))/scale)**2).mean(axis=1))
            peers = peers[distance <= 2.5]
            rec['peer_rows'] = len(peers)
            # Avoid far extrapolation of any demand/context feature.
            outside = ((row[CONTEXT].astype(float) < stable[CONTEXT].min()-scale) |
                       (row[CONTEXT].astype(float) > stable[CONTEXT].max()+scale)).any()
            if outside or len(peers) < settings.min_peer_rows or peers.team.nunique() < settings.min_peer_teams:
                rec['reason'] = '조건이 가까운 비교사례 부족 또는 학습범위 밖'
            else:
                estimates = []
                for q in [.1, .5, .9]:
                    estimator = make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, alpha=.05, solver='highs'))
                    estimator.fit(stable[CONTEXT], stable.headcount)
                    estimates.append(float(estimator.predict(row[CONTEXT].to_frame().T.astype(float))[0]))
                if not np.isfinite(estimates).all() or min(estimates) <= 0:
                    rec['reason'] = '유효한 양의 인원범위를 추정하지 못함'
                else:
                    ordered = np.sort(estimates)
                    rec.update(status='경험적 비교범위', reason='일정·OT·휴가 기준; 품질 및 업무범위 미검증',
                               p10=ordered[0], p50=ordered[1], p90=ordered[2],
                               quantile_rearranged=bool(np.any(np.diff(estimates) < 0)))
                    rec.update(dict(zip(['peer_p10', 'peer_p50', 'peer_p90'], peers.headcount.quantile([.1, .5, .9]))))
        results.append(rec)
    return pd.DataFrame(results)


def peer_backtest(panel, settings=Settings()):
    records = []
    for month in sorted(panel.month.unique()):
        result = peer_staffing(panel, month, settings)
        actual = panel[panel.month == month].copy()
        actual['stable_actual'] = stable_mask(actual, settings)
        result = result.merge(actual[['team', 'headcount', 'stable_actual']], on='team', validate='one_to_one')
        result['evaluated'] = (result.status == '경험적 비교범위') & result.stable_actual
        result['covered'] = np.where(result.evaluated, (result.headcount >= result.p10) & (result.headcount <= result.p90), np.nan)
        result['interval_width'] = result.p90-result.p10
        records.append(result)
    return pd.concat(records, ignore_index=True)


def ot_examples(panel):
    source = panel.copy()
    source['source_month'] = source.month
    source['target_month'] = (pd.PeriodIndex(source.month, freq='M')+1).astype(str)
    source['entity'] = source.team
    target = panel[['month', 'team', 'ot_mean']].rename(columns={'month':'target_month', 'ot_mean':'actual'})
    # Exact next calendar month, no shift across missing months.
    return source.merge(target, on=['target_month', 'team'], how='left', validate='one_to_one')


def make_estimator(name, classification):
    if name == 'Regression':
        model = LogisticRegression(C=1., max_iter=1000, random_state=42) if classification else Ridge(alpha=10.)
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), model)
    if name == 'Random Forest':
        cls = RandomForestClassifier if classification else RandomForestRegressor
        model = cls(n_estimators=80, max_depth=5, min_samples_leaf=3, random_state=42, n_jobs=2)
    elif name == 'XGBoost':
        cls = XGBClassifier if classification else XGBRegressor
        model = cls(n_estimators=80, max_depth=3, learning_rate=.05, random_state=42, n_jobs=2,
                    objective='binary:logistic' if classification else 'reg:squarederror')
    else:
        raise ValueError(f'알 수 없는 알고리즘: {name}')
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), model)


def choose_algorithm(scores, before_month, active, settings):
    """Select only from previously realized, common OOS forecast months."""
    if scores.empty:
        return 'Baseline', '과거 공통 검증 부족', None, {}
    eligible = scores[(scores.target_month < before_month) & (scores.status == 'ok')].copy()
    complete = eligible.groupby('target_month').algorithm.nunique()
    months = sorted(complete[complete == len(ALGORITHMS)].index)[-settings.selection_months:]
    if len(months) < settings.min_validation_months:
        return 'Baseline', '과거 공통 검증 부족', max(months) if months else None, {}
    recent = eligible[eligible.target_month.isin(months)]
    # Each algorithm was scored on the same rows. Pool errors, not unequal monthly means.
    means = {name: float(part.error_sum.sum()/part.n.sum()) for name, part in recent.groupby('algorithm')}
    best = min(ALGORITHMS, key=lambda name: means[name])
    if means[active]-means[best] > max(1e-12, settings.min_improvement):
        active = best
    return active, '이전 공통 평가창 기준', max(months), means


def risk_tournament(examples, task='ot', settings=Settings()):
    """Expanding-window OOS forecasts + next-month winner, separately by family.

    Every eligible month refits *all* algorithms. Winner uses only earlier OOS
    outcomes; current outcomes never choose the forecast already being issued.
    Latest unlabeled rows are forecast too. Gate scores are Brier, OT scores MAE.
    """
    if task not in ['ot', 'gate']:
        raise ValueError('task는 ot 또는 gate입니다.')
    if examples.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    classification = task == 'gate'
    features = GATE_FEATURES if classification else RISK_FEATURES
    required = features + ['source_month', 'target_month', 'entity', 'development_type', 'actual']
    if set(required)-set(examples.columns):
        raise ValueError(f'모델 입력 열 누락: {set(required)-set(examples.columns)}')
    if examples.duplicated(['target_month', 'entity']).any():
        raise ValueError('같은 대상·월의 중복 예측 행')
    expected_month = (pd.PeriodIndex(examples.source_month, freq='M')+1).astype(str)
    if not np.array_equal(expected_month, examples.target_month):
        raise ValueError('예측은 정확히 다음 달이어야 합니다.')
    known = examples.actual.dropna()
    if not np.isfinite(known).all() or (known < 0).any() or (classification and not known.isin([0, 1]).all()):
        raise ValueError('유효하지 않은 관측 정답')
    predictions, evaluations, selections = [], [], []
    for family, group in examples.groupby('development_type'):
        score_history = []
        active = 'Baseline'
        for month in sorted(group.target_month.unique()):
            test = group[group.target_month == month].copy()
            train = group[(group.target_month < month) & group.actual.notna()]
            active, reason, selection_end, prior_scores = choose_algorithm(pd.DataFrame(score_history), month, active, settings)
            enough = len(train) >= settings.min_train_rows and train.target_month.nunique() >= settings.min_history_months
            both_classes = not classification or train.actual.nunique() == 2
            available, unavailable = {}, {}
            if classification:
                if len(train):
                    available['Baseline'] = np.full(len(test), train.actual.mean())
                else:
                    unavailable['Baseline'] = '관측 Gate 결과 없음'
            else:
                available['Baseline'] = test.ot_mean.to_numpy(float)
            for name in ALGORITHMS[1:]:
                if not enough or not both_classes:
                    unavailable[name] = '학습기간·관측수 부족' if not enough else 'Gate 지연/준수 두 종류의 관측 필요'
                    continue
                model = make_estimator(name, classification)
                model.fit(train[features], train.actual.astype(int) if classification else train.actual)
                available[name] = (np.clip(model.predict_proba(test[features])[:, 1], 0, 1) if classification
                                   else np.maximum(0, model.predict(test[features])))
            if active not in available:
                active = 'Baseline' if 'Baseline' in available else None
                reason = '선택 알고리즘 사용 불가; 기준모델 또는 판단 유보'
            selections.append(dict(task=task, development_type=family, target_month=month,
                selected_algorithm=active or '판단 유보', selection_through=selection_end,
                reason=reason, prior_scores=json.dumps({k:round(v, 12) for k,v in prior_scores.items()}, sort_keys=True), training_rows=len(train),
                training_through=train.target_month.max() if len(train) else None))
            for name in ALGORITHMS:
                observed = test.actual.notna().to_numpy()
                status = 'ok' if name in available else unavailable[name]
                n, error_sum, score = 0, np.nan, np.nan
                if name in available:
                    values = available[name]
                    for pos, (_, row) in enumerate(test.iterrows()):
                        predictions.append(dict(task=task, development_type=family, target_month=month,
                            source_month=row.source_month, entity=row.entity, algorithm=name,
                            prediction=float(values[pos]), actual=row.actual, selected=name == active,
                            training_through=train.target_month.max() if len(train) else None,
                            selection_through=selection_end))
                    n = int(observed.sum())
                    if n:
                        error = values[observed]-test.actual.to_numpy(float)[observed]
                        error_sum = float(np.square(error).sum() if classification else np.abs(error).sum())
                        score = error_sum/n
                    else:
                        status = '평가 대기'
                evaluation = dict(task=task, development_type=family, target_month=month, algorithm=name,
                                  metric='Brier' if classification else 'MAE', score=score,
                                  error_sum=error_sum, n=n, status=status)
                evaluations.append(evaluation)
                score_history.append(evaluation)
            if active is None:
                active = 'Baseline'
    return pd.DataFrame(predictions), pd.DataFrame(evaluations), pd.DataFrame(selections)
