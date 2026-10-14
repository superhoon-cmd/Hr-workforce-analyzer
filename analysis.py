"""Observed workload signals and chronological forecasts of measurable OT."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sample_data import COLUMNS
from pathlib import Path
import json
import joblib

PROJECT = '개발Project명'
MONTH = '기준월'
NUMERIC = COLUMNS[5:9] + COLUMNS[10:]
FEATURES = ['current_fte', 'ot_per_fte', 'previous_avg_ot', 'previous_available',
            'ot_change', 'difficulty', 'capability', 'gate', 'centrality', 'sent',
            'received', 'completed_share', 'max_contribution_share', 'contribution_hhi']
METHODS = ['이전월 유지', 'Random Forest', 'XGBoost']


def validate_csv(df):
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'필수 열 누락: {", ".join(sorted(missing))}')
    df = df[COLUMNS].copy()
    if df.empty:
        raise ValueError('CSV에 데이터가 없습니다.')
    for col in ['사번', '이름', MONTH, '팀', PROJECT]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col}: 빈 값은 허용되지 않습니다.')
    if not df[MONTH].astype(str).str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():
        raise ValueError('기준월은 yyyy-mm 형식이어야 합니다.')
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not np.isfinite(df[col]).all():
            raise ValueError(f'{col}: 유효한 숫자가 필요합니다.')
    for col in ['개발Project난이도', '직무역량Level평균', '개발Gate 단계']:
        if not df[col].between(1, 5).all():
            raise ValueError(f'{col}: 1~5 범위가 필요합니다.')
    if not df['중앙성(network분석결과)'].between(0, 1).all():
        raise ValueError('중앙성은 0~1 범위가 필요합니다.')
    if not df['업무비중(%)'].between(0, 100, inclusive='right').all():
        raise ValueError('업무비중은 0 초과 100 이하이어야 합니다.')
    if (df[['총OT(초과근로)', '메일 발신량', '메일 수신량']] < 0).any().any():
        raise ValueError('OT와 메일량은 음수일 수 없습니다.')
    if df.duplicated(['사번', MONTH, PROJECT]).any():
        raise ValueError('사번·기준월·프로젝트 중복 행이 있습니다.')
    grouped = df.groupby(['사번', MONTH])
    if not np.allclose(grouped['업무비중(%)'].sum(), 100, atol=0.01):
        raise ValueError('각 직원의 월별 업무비중 합계는 100%여야 합니다.')
    # Employee-level measurements are repeated on assignment rows, never additive.
    for col in ['이름', '팀', '직무역량Level평균'] + COLUMNS[10:]:
        if (grouped[col].nunique() > 1).any():
            raise ValueError(f'{col}: 같은 직원·월의 프로젝트 행에는 동일한 값을 넣으세요.')
    dates = pd.to_datetime(df['Gate 완료일'].replace('', np.nan), errors='coerce')
    provided = df['Gate 완료일'].notna() & df['Gate 완료일'].astype(str).str.strip().ne('')
    if (provided & dates.isna()).any():
        raise ValueError('Gate 완료일은 유효한 날짜 또는 빈 값이어야 합니다.')
    month_end = pd.to_datetime(df[MONTH] + '-01') + pd.offsets.MonthEnd(0)
    if (dates > month_end).any():
        raise ValueError('Gate 완료일에 기준월 이후의 미래 실적을 넣을 수 없습니다.')
    df['Gate 완료일'] = dates
    return df


def aggregate_projects(df):
    df = df.copy()
    df['share'] = df['업무비중(%)'] / 100
    df['allocated_ot'] = df['총OT(초과근로)'] * df['share']
    mapping = {'difficulty': '개발Project난이도', 'capability': '직무역량Level평균',
               'gate': '개발Gate 단계', 'centrality': '중앙성(network분석결과)',
               'sent': '메일 발신량', 'received': '메일 수신량'}
    for target, source in mapping.items():
        df[target] = df[source] * df['share']
    df['completed_share'] = df['Gate 완료일'].notna() * df['share']
    df['share_squared'] = df['share'] ** 2
    out = df.groupby([MONTH, PROJECT], as_index=False).agg(
        current_fte=('share', 'sum'), ot_hours=('allocated_ot', 'sum'),
        participants=('사번', 'nunique'), max_share=('share', 'max'),
        squared_shares=('share_squared', 'sum'),
        **{k: (k, 'sum') for k in mapping}, completed_share=('completed_share', 'sum'))
    for col in list(mapping) + ['completed_share']:
        out[col] /= out['current_fte']
    out['ot_per_fte'] = out['ot_hours'] / out['current_fte']
    out['max_contribution_share'] = out['max_share'] / out['current_fte']
    out['contribution_hhi'] = out['squared_shares'] / out['current_fte'] ** 2
    history = out[[MONTH, PROJECT, 'ot_per_fte']].copy()
    history[MONTH] = (pd.PeriodIndex(history[MONTH], freq='M') + 1).astype(str)
    history = history.rename(columns={'ot_per_fte': 'previous_avg_ot'})
    out = out.merge(history, on=[MONTH, PROJECT], how='left', validate='one_to_one')
    out['previous_available'] = out['previous_avg_ot'].notna().astype(int)
    out['ot_change'] = out['ot_per_fte'] - out['previous_avg_ot']
    return out.drop(columns=['max_share', 'squared_shares'])


def diagnose_projects(projects, min_peers=3, high_ot=20.0, low_ot=10.0):
    """Explain relative workload signals, never infer a headcount ground truth."""
    records = []
    lookup = projects.set_index([MONTH, PROJECT])['ot_per_fte']
    for _, row in projects.iterrows():
        peers = projects[(projects[MONTH] == row[MONTH]) &
                         (projects[PROJECT] != row[PROJECT]) &
                         ((projects.difficulty - row.difficulty).abs() <= 1) &
                         ((projects.gate - row.gate).abs() <= 1) &
                         ((projects.capability - row.capability).abs() <= 0.75)]
        peer_ot = peers.ot_per_fte.median() if len(peers) >= min_peers else np.nan
        peer_fte = peers.current_fte.median() if len(peers) >= min_peers else np.nan
        month = pd.Period(row[MONTH], freq='M')
        history = [lookup.get((str(month - lag), row[PROJECT]), np.nan) for lag in [2, 1, 0]]
        sustained = all(np.isfinite(v) and v >= high_ot for v in history)
        rising = all(np.isfinite(v) for v in history) and history[1] > history[0] + 2 and history[2] > history[1] + 2
        status = '판단 유보'
        if len(peers) < min_peers:
            reason = f'유사 프로젝트 {len(peers)}개: 최소 {min_peers}개 미만으로 비교 근거 부족'
        elif (row.ot_per_fte >= high_ot and row.ot_per_fte >= max(peer_ot * 1.25, peer_ot + 5)
              and row.current_fte <= peer_fte * 0.9 and (sustained or rising)):
            status = '부족 가능성'
            reason = '유사 프로젝트보다 투입 FTE가 적고 OT가 높으며, 3개월 고부담 또는 상승 추세'
        elif (peer_ot > 0 and row.ot_per_fte <= low_ot and row.ot_per_fte <= peer_ot * 0.75
              and row.current_fte >= peer_fte * 1.1):
            status = '여유 가능성'
            reason = '유사 프로젝트보다 투입 FTE가 많고 OT가 낮음; 납기·품질·산출물 확인 필요'
        elif row.ot_per_fte >= high_ot and (sustained or rising):
            reason = 'OT 부담 지속: 인원 부족·재작업·일정 집중·운영 비효율 구분 필요'
        else:
            reason = '인력 부족·여유를 구분할 복합 신호 부족; 정상 인원 판정은 아님'
        if row.max_contribution_share >= 0.3:
            reason += '; 한 참여자의 기여 비중 30% 이상으로 병목 가능성 확인'
        records.append({MONTH: row[MONTH], PROJECT: row[PROJECT], '판단': status, '근거': reason,
                        '유사 프로젝트 수': len(peers), '유사 평균 OT 중앙값(시간/FTE)': peer_ot,
                        '유사 투입 FTE 중앙값': peer_fte, '3개월 OT 고부담': sustained,
                        '3개월 OT 상승': rising})
    return projects.merge(pd.DataFrame(records), on=[MONTH, PROJECT], validate='one_to_one')


def should_replace_model(active_mae, candidate_mae, min_improvement=0.0):
    """A strictly better result on the same observed month is required."""
    return active_mae - candidate_mae > max(min_improvement, 1e-12)


def walk_forward(projects, min_improvement=0.0, model_dir=None):
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError('최소 개선 폭은 0 이상의 유한한 값이어야 합니다.')
    previous = projects.copy()
    previous['source_month'] = previous[MONTH]
    previous[MONTH] = (pd.PeriodIndex(previous[MONTH], freq='M') + 1).astype(str)
    targets = projects[[MONTH, PROJECT, 'ot_per_fte']].rename(columns={'ot_per_fte': 'actual_ot'})
    pairs = previous.merge(targets, on=[MONTH, PROJECT], validate='one_to_one')
    predictions, metrics = [], []
    active_models, active_metadata = {}, {}
    for month in sorted(projects[MONTH].unique()):
        if int(month[-2:]) not in [7, 8, 9]:
            continue
        year = month[:4]
        # Every usable historical pair is retained, including earlier years/months.
        train = pairs[pairs[MONTH] < month]
        test = pairs[pairs[MONTH] == month]
        expected = [f'{year}-{m:02d}' for m in range(2, int(month[-2:]))]
        if not set(expected).issubset(set(projects[MONTH])):
            raise ValueError(f'{month} 예측에는 2월부터 직전월까지 연속 데이터가 필요합니다.')
        if train.empty or test.empty:
            raise ValueError(f'{month}: 연속 월 학습·평가 프로젝트가 없습니다.')
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                                                   random_state=42, n_jobs=2),
            'XGBoost': XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05,
                                    objective='reg:squarederror', random_state=42, n_jobs=2)}
        estimates = {'이전월 유지': test.ot_per_fte.to_numpy()}
        decisions = {}
        for name, candidate in models.items():
            # y is observed next-month OT, not a formula-generated staffing label.
            candidate.fit(train[FEATURES].fillna(0), train.actual_ot)
            candidate_estimate = np.maximum(0, candidate.predict(test[FEATURES].fillna(0)))
            initial = name not in active_models
            if initial:
                active_models[name] = candidate
                active_metadata[name] = {'training_start': train.source_month.min(),
                                         'training_end': train[MONTH].max(), 'rows': len(train)}
            # Issue the forecast using the already selected model, before seeing y.
            estimates[name] = np.maximum(0, active_models[name].predict(test[FEATURES].fillna(0)))
            active_info = active_metadata[name].copy()
            active_mae = mean_absolute_error(test.actual_ot, estimates[name])
            candidate_mae = mean_absolute_error(test.actual_ot, candidate_estimate)
            replace = not initial and should_replace_model(active_mae, candidate_mae, min_improvement)
            # Outcome is observed now; promotion only affects the next forecast.
            if replace:
                active_models[name] = candidate
                active_metadata[name] = {'training_start': train.source_month.min(),
                                         'training_end': train[MONTH].max(), 'rows': len(train)}
            decisions[name] = {'candidate_prediction': candidate_estimate,
                               'active_info': active_info, 'candidate_mae': candidate_mae,
                               'action': '초기 등록' if initial else ('교체' if replace else '유지'),
                               'next_training_end': active_metadata[name]['training_end']}
        for name, estimate in estimates.items():
            result = test[[MONTH, PROJECT, 'source_month', 'actual_ot']].copy()
            result['model'] = name
            result['prediction'] = estimate
            decision = decisions.get(name)
            result['training_target_end'] = decision['active_info']['training_end'] if decision else None
            result['candidate_prediction'] = decision['candidate_prediction'] if decision else estimate
            predictions.append(result)
            metrics.append({MONTH: month, 'model': name,
                            'MAE(시간/FTE)': mean_absolute_error(test.actual_ot, estimate),
                            'RMSE(시간/FTE)': np.sqrt(mean_squared_error(test.actual_ot, estimate)),
                            '학습 건수': decision['active_info']['rows'] if decision else 0,
                            '누적 후보 학습 건수': len(train) if decision else 0,
                            '누적 학습 시작 월': train.source_month.min() if decision else None,
                            '후보 학습 정답 마지막 월': train[MONTH].max() if decision else None,
                            '후보 MAE(시간/FTE)': decision['candidate_mae'] if decision else np.nan,
                            '모델 결정': decision['action'] if decision else '기준 모델',
                            '결정 적용 월': str(pd.Period(month, freq='M') + 1) if decision else None,
                            '다음 모델 학습 마지막 월': decision['next_training_end'] if decision else None,
                            '평가 건수': len(test),
                            '학습 정답 마지막 월': decision['active_info']['training_end'] if decision else None})
    if not predictions:
        raise ValueError('2~6월 학습과 7월 평가를 위한 데이터가 필요합니다.')
    if model_dir is not None:
        directory = Path(model_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in active_models.items():
            joblib.dump(model, directory / ('random_forest.joblib' if name == 'Random Forest' else 'xgboost.joblib'))
        (directory / 'metadata.json').write_text(json.dumps(
            {'models': active_metadata, 'features': FEATURES, 'target': 'next-month observed OT hours/FTE',
             'evaluated_through': max(m[MONTH] for m in metrics), 'min_improvement': min_improvement},
            ensure_ascii=False, indent=2))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def project_comparison(predictions, month):
    selected = predictions[predictions[MONTH] == month]
    pivot = selected.pivot(index=PROJECT, columns='model', values='prediction')[METHODS]
    actual = selected.drop_duplicates(PROJECT).set_index(PROJECT)
    pivot['관측 OT(시간/FTE)'] = actual.actual_ot
    pivot['ML 평균 OT(시간/FTE)'] = pivot[['Random Forest', 'XGBoost']].mean(axis=1)
    pivot['모델 차이(시간/FTE)'] = (pivot['Random Forest'] - pivot['XGBoost']).abs()
    return pivot.reset_index()


def team_summary(df, diagnosis, month):
    """Current-month project exposure, not inferred headcount or team verdicts."""
    rows = df[df[MONTH] == month].copy()
    rows['투입 FTE'] = rows['업무비중(%)'] / 100
    rows['배분 OT(시간)'] = rows['총OT(초과근로)'] * rows['투입 FTE']
    rows = rows.merge(diagnosis.loc[diagnosis[MONTH] == month, [PROJECT, '판단']],
                      on=PROJECT, how='left', validate='many_to_one')
    out = rows.groupby('팀').agg(현재인원=('사번', 'nunique'), 투입FTE=('투입 FTE', 'sum'),
                                  총OT=('배분 OT(시간)', 'sum'))
    out = out.rename(columns={'현재인원': '현재 인원', '투입FTE': '투입 FTE', '총OT': '월 OT(시간)'})
    out['평균 OT(시간/FTE)'] = out['월 OT(시간)'] / out['투입 FTE']
    for status in ['부족 가능성', '여유 가능성', '판단 유보']:
        exposed = rows[rows['판단'] == status].groupby('팀')['투입 FTE'].sum()
        out[f'{status} 프로젝트 기여(%)'] = exposed.reindex(out.index).fillna(0) / out['투입 FTE'] * 100
    return out.reset_index()
