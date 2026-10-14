import numpy as np
import pandas as pd
import pytest
from sample_data import generate_sample
from analysis import METHODS, should_replace_model, validate_csv, aggregate_projects, diagnose_projects, walk_forward, project_comparison, team_summary


@pytest.fixture(scope='module')
def frame():
    return validate_csv(generate_sample())


def test_sample_and_ot_allocation(frame):
    projects = aggregate_projects(frame)
    assert len(frame) == 12000
    assert frame['사번'].nunique() == 500
    assert frame['팀'].nunique() == 20
    assert frame['개발Project명'].nunique() == 30
    assert np.allclose(projects.groupby('기준월').current_fte.sum(), 500)
    observed = frame.drop_duplicates(['사번', '기준월']).groupby('기준월')['총OT(초과근로)'].sum()
    assert np.allclose(projects.groupby('기준월').ot_hours.sum(), observed)
    assert np.allclose(projects.ot_per_fte, projects.ot_hours / projects.current_fte)
    assert not {'required_fte', 'base_required_fte', 'ot_intercept_fte'} & set(projects.columns)


def test_previous_ot_uses_exact_calendar_and_missing_flag(frame):
    projects = aggregate_projects(frame)
    june = projects[projects['기준월'] == '2026-06'].set_index('개발Project명')
    july = projects[projects['기준월'] == '2026-07'].set_index('개발Project명')
    assert np.allclose(july.previous_avg_ot, june.ot_per_fte)
    feb = projects[projects['기준월'] == '2026-02']
    assert feb.previous_avg_ot.isna().all()
    assert (feb.previous_available == 0).all()
    missing = aggregate_projects(frame[frame['기준월'] != '2026-06'])
    assert missing.loc[missing['기준월'] == '2026-07', 'previous_avg_ot'].isna().all()


def signal_fixture():
    records = []
    for month in ['2026-05', '2026-06', '2026-07']:
        for name, fte, ot in [('LOW', 5, 40), ('HIGH', 20, 5), ('P1', 10, 20), ('P2', 10, 20), ('P3', 10, 20), ('P4', 10, 20)]:
            records.append({'기준월': month, '개발Project명': name, 'current_fte': fte,
                            'ot_per_fte': ot, 'difficulty': 3, 'gate': 2, 'capability': 3,
                            'max_contribution_share': 0.1})
    return pd.DataFrame(records)


def test_signal_categories_and_explanations():
    result = diagnose_projects(signal_fixture())
    july = result[result['기준월'] == '2026-07'].set_index('개발Project명')
    assert july.loc['LOW', '판단'] == '부족 가능성'
    assert july.loc['HIGH', '판단'] == '여유 가능성'
    assert july.loc['P1', '판단'] == '판단 유보'
    assert '납기·품질' in july.loc['HIGH', '근거']
    # Not enough history to label a single high-OT month as a shortage.
    assert result.loc[(result['기준월'] == '2026-05') & (result['개발Project명'] == 'LOW'), '판단'].iloc[0] == '판단 유보'


def test_signal_abstains_without_peers_or_contiguous_history():
    data = signal_fixture()
    no_peers = data[data['개발Project명'] == 'LOW']
    assert (diagnose_projects(no_peers)['판단'] == '판단 유보').all()
    missing = data[data['기준월'] != '2026-06']
    result = diagnose_projects(missing)
    assert result.loc[(result['기준월'] == '2026-07') & (result['개발Project명'] == 'LOW'), '판단'].iloc[0] == '판단 유보'
    # Low OT alone is not enough: remove HIGH's elevated staffing.
    data.loc[data['개발Project명'] == 'HIGH', 'current_fte'] = 10
    assert (diagnose_projects(data).loc[lambda x: x['개발Project명'] == 'HIGH', '판단'] == '판단 유보').all()


def test_walk_forward_uses_observed_ot_and_expands_training(frame):
    projects = aggregate_projects(frame)
    predictions, metrics = walk_forward(projects)
    assert len(predictions) == 270 and len(metrics) == 9
    assert set(predictions.model) == set(METHODS)
    assert set(metrics['평가 건수']) == {30}
    assert list(metrics[metrics.model == 'XGBoost']['누적 후보 학습 건수']) == [120, 150, 180]
    learned = predictions[predictions.model != '이전월 유지']
    assert (learned.training_target_end < learned['기준월']).all()
    assert (predictions.source_month < predictions['기준월']).all()
    observed = predictions.merge(projects[['기준월', '개발Project명', 'ot_per_fte']], on=['기준월', '개발Project명'])
    assert np.allclose(observed.actual_ot, observed.ot_per_fte)
    prior = predictions.merge(projects[['기준월', '개발Project명', 'ot_per_fte']],
                              left_on=['source_month', '개발Project명'], right_on=['기준월', '개발Project명'])
    baseline = prior[prior.model == '이전월 유지']
    assert np.allclose(baseline.prediction, baseline.ot_per_fte)


def test_no_future_leakage_in_forecast_or_signals(frame):
    baseline_projects = aggregate_projects(frame)
    baseline, _ = walk_forward(baseline_projects)
    changed = frame.copy()
    changed.loc[changed['기준월'] >= '2026-07', '총OT(초과근로)'] += 100
    changed_projects = aggregate_projects(changed)
    revised, _ = walk_forward(changed_projects)
    assert np.allclose(baseline.loc[baseline['기준월'] == '2026-07', 'prediction'], revised.loc[revised['기준월'] == '2026-07', 'prediction'])
    assert not np.allclose(baseline.loc[baseline['기준월'] == '2026-08', 'prediction'], revised.loc[revised['기준월'] == '2026-08', 'prediction'])
    before = diagnose_projects(baseline_projects)
    after = diagnose_projects(changed_projects)
    pd.testing.assert_frame_equal(before[before['기준월'] < '2026-07'], after[after['기준월'] < '2026-07'])


def test_team_workload_conservation_and_no_staffing_claims(frame):
    projects = aggregate_projects(frame)
    signals = diagnose_projects(projects)
    teams = team_summary(frame, signals, '2026-09')
    assert len(teams) == 20 and teams['현재 인원'].sum() == 500
    assert np.isclose(teams['투입 FTE'].sum(), 500)
    assert np.isclose(teams['월 OT(시간)'].sum(), projects.loc[projects['기준월'] == '2026-09', 'ot_hours'].sum())
    shares = [c for c in teams.columns if c.endswith('기여(%)')]
    assert np.allclose(teams[shares].sum(axis=1), 100)
    predictions, _ = walk_forward(projects)
    result = project_comparison(predictions, '2026-09')
    assert not {'부족 FTE', '평균 추정 FTE', '관측 대리 지표 FTE'} & set(result.columns)
    assert not {'부족 FTE', '추정 FTE'} & set(teams.columns)


def test_replacement_requires_strict_same_month_improvement():
    assert should_replace_model(4, 3)
    assert not should_replace_model(4, 4)
    assert not should_replace_model(4, 5)
    assert not should_replace_model(4, 3.9, min_improvement=0.2)


def test_candidate_promotion_only_affects_following_month(frame, monkeypatch):
    import analysis

    class ControlledRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            self.offset = {120: 2, 150: 1, 180: 3}[len(x)]

        def predict(self, x):
            return np.full(len(x), 10 + self.offset)

    monkeypatch.setattr(analysis, 'RandomForestRegressor', ControlledRegressor)
    monkeypatch.setattr(analysis, 'XGBRegressor', ControlledRegressor)
    data = frame.copy()
    data['총OT(초과근로)'] = 10.0
    projects = aggregate_projects(data)
    predictions, metrics = walk_forward(projects)
    rf = metrics[metrics.model == 'Random Forest']
    assert list(rf['모델 결정']) == ['초기 등록', '교체', '유지']
    assert list(rf['학습 정답 마지막 월']) == ['2026-06', '2026-06', '2026-07']
    assert list(rf['다음 모델 학습 마지막 월']) == ['2026-06', '2026-07', '2026-07']
    for month, expected in [('2026-07', 12), ('2026-08', 12), ('2026-09', 11)]:
        rows = predictions[(predictions.model == 'Random Forest') & (predictions['기준월'] == month)]
        assert np.allclose(rows.prediction, expected)
    # A larger required improvement freezes the original fitted model.
    frozen, log = walk_forward(projects, min_improvement=10)
    assert np.allclose(frozen.loc[frozen.model == 'Random Forest', 'prediction'], 12)
    assert list(log.loc[log.model == 'Random Forest', '모델 결정']) == ['초기 등록', '유지', '유지']


def test_cumulative_training_includes_data_before_february(frame):
    january = frame[frame['기준월'] == '2026-02'].copy()
    january['기준월'] = '2026-01'
    projects = aggregate_projects(pd.concat([january, frame], ignore_index=True))
    _, metrics = walk_forward(projects)
    rf = metrics[metrics.model == 'Random Forest']
    assert list(rf['누적 후보 학습 건수']) == [150, 180, 210]
    assert (rf['누적 학습 시작 월'] == '2026-01').all()


def test_selected_models_are_saved_with_their_training_range(frame, tmp_path):
    import json
    import joblib
    projects = aggregate_projects(frame)
    predictions, _ = walk_forward(projects, min_improvement=100, model_dir=tmp_path)
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    assert metadata['models']['Random Forest']['training_end'] == '2026-06'
    assert (tmp_path / 'xgboost.joblib').is_file()
    saved = joblib.load(tmp_path / 'random_forest.joblib')
    september = predictions[(predictions.model == 'Random Forest') & (predictions['기준월'] == '2026-09')]
    august = projects[projects['기준월'] == '2026-08'].set_index('개발Project명').loc[september['개발Project명']]
    assert np.allclose(saved.predict(august[metadata['features']].fillna(0)), september.prediction)


@pytest.mark.parametrize('change', ['share', 'future_date', 'duplicate', 'negative_ot', 'missing'])
def test_invalid_csv(frame, change):
    bad = frame.copy()
    if change == 'share':
        bad.loc[0, '업무비중(%)'] = 10
    elif change == 'future_date':
        bad.loc[0, 'Gate 완료일'] = '2026-12-01'
    elif change == 'duplicate':
        bad = pd.concat([bad, bad.iloc[[0]]])
    elif change == 'negative_ot':
        bad.loc[0, '총OT(초과근로)'] = -1
    else:
        bad = bad.drop(columns=['팀'])
    with pytest.raises(ValueError):
        validate_csv(bad)
