import numpy as np
import pandas as pd
import pytest
from workforce_data import build_panels, generate_snapshots, validate_snapshots, CONTEXT
from workforce_models import (Settings, ALGORITHMS, normalize_ot, peer_staffing,
    peer_backtest, stable_mask, ot_examples, risk_tournament, choose_algorithm)


@pytest.fixture(scope='module')
def snapshots():
    return generate_snapshots(months=12)


@pytest.fixture(scope='module')
def panels(snapshots):
    return build_panels(*snapshots)


def test_hours_allocation_and_exact_calendar(panels, snapshots):
    panel, _ = panels
    e, _, _ = snapshots
    assert np.allclose(panel.groupby('month').ot_total.sum(), e.groupby('month').ot_hours.sum())
    assert np.allclose(panel.groupby('month').available_hours.sum(), (e.normal_hours-e.leave_hours).groupby(e.month).sum())
    assert panel[panel.month == '2025-01'].joiners.isna().all()
    assert (panel[panel.month == '2025-02'].joiners == 0).all()
    missing = panel[panel.month != '2025-06']
    pairs = ot_examples(missing)
    assert pairs[pairs.source_month == '2025-05'].actual.isna().all()


def test_ot_normalization_uses_leave_once(panels):
    p = panels[0].iloc[[0]].copy()
    p[['headcount','normal_hours','available_hours','planned_leave_hours','ot_total']] = [20,3200,3040,160,600]
    out = normalize_ot(p, 10)
    assert out.normalized_staffing.iloc[0] == pytest.approx(3640/162)


@pytest.mark.parametrize('case', ['share', 'duplicate', 'future', 'original', 'leave', 'missing_project'])
def test_input_contract(snapshots, case):
    e,a,g = [x.copy() for x in snapshots]
    if case == 'share': a.loc[0,'share'] = .5
    elif case == 'duplicate': e = pd.concat([e,e.iloc[[0]]])
    elif case == 'future': g.loc[0,'completed_at'] = pd.Timestamp('2030-01-01')
    elif case == 'original': g.loc[0,'original_due'] += pd.Timedelta(days=1); g.loc[1,'original_due'] += pd.Timedelta(days=1)
    elif case == 'leave': e.loc[0,'leave_hours'] = 999
    else: g = g[g.project_id != 'P00']
    with pytest.raises(ValueError): validate_snapshots(e,a,g)


def test_gate_deadline_frozen_and_missing_outcome_unknown(snapshots):
    e,a,g = [x.copy() for x in snapshots]
    _, before = build_panels(e,a,g)
    mask = (g.month == '2025-02') & (g.gate_id == 'G01')
    g.loc[mask,'planned_due'] += pd.Timedelta(days=30)
    _, after = build_panels(e,a,g)
    cols = ['entity','actual','frozen_due']
    pd.testing.assert_frame_equal(before[before.target_month=='2025-02'][cols], after[after.target_month=='2025-02'][cols])
    assert not after.entity.duplicated().any()
    # Remove one outcome snapshot, while other Gate row still covers the project.
    g = g[~((g.month=='2025-02') & (g.gate_id=='G01') & (g.project_id=='P00'))]
    _, missing = build_panels(e,a,g)
    assert missing.loc[missing.entity=='P00/G01','actual'].isna().all()


def test_shared_project_gate_not_duplicated(snapshots):
    e,a,g = [x.copy() for x in snapshots]
    # Split one project's staff into two teams, preserve the project ID.
    e.loc[(e.team=='Cell_00') & e.employee_id.str.endswith('000'), 'team'] = 'Cell_support'
    _, gates = build_panels(e,a,g)
    assert not gates.entity.duplicated().any()


def test_peer_quantiles_no_current_headcount_or_future_leakage(panels):
    p = panels[0]
    settings = Settings(min_peer_rows=10)
    before = peer_staffing(p,'2025-11',settings)
    assert (before.status == '경험적 비교범위').any()
    altered = p.copy()
    altered.loc[altered.month >= '2025-11', ['headcount','ot_total','ot_p90','gate_misses']] = 999
    after = peer_staffing(altered,'2025-11',settings)
    pd.testing.assert_frame_equal(before, after)
    available = before[before.p50.notna()]
    assert (available.p10 <= available.p50).all() and (available.p50 <= available.p90).all()
    assert not set(['headcount','available_hours','ot_total','ot_mean']) & set(CONTEXT)


def test_peer_abstains_without_gate_results_and_type_pooling(panels):
    p = panels[0].copy()
    p.loc[p.development_type=='Cell','gates_observed'] = 0
    r = peer_staffing(p,'2025-12')
    assert r.loc[r.development_type=='Cell','p50'].isna().all()
    assert r.loc[r.development_type=='Cell','stable_rows'].eq(0).all()


def test_peer_backtest_does_not_evaluate_unknown_outcomes(panels):
    p = panels[0].copy()
    p.loc[p.month=='2025-12','gates_observed'] = 0
    r = peer_backtest(p,Settings(min_peer_rows=10))
    assert not r.loc[r.month=='2025-12','evaluated'].any()


def test_selection_uses_common_prior_months_and_margin():
    records=[]
    for month in ['2025-01','2025-02','2025-03']:
        for name,err in zip(ALGORITHMS,[4.,3.,2.,1.]):
            records.append(dict(target_month=month,algorithm=name,error_sum=err*5,n=5,status='ok'))
    scores=pd.DataFrame(records)
    selected,_,through,_=choose_algorithm(scores,'2025-03','Baseline',Settings())
    assert selected=='XGBoost' and through=='2025-02'
    # The current evaluation cannot change the current selection.
    scores.loc[(scores.target_month=='2025-03') & (scores.algorithm=='Baseline'),'error_sum']=0
    assert choose_algorithm(scores,'2025-03','Baseline',Settings())[0]=='XGBoost'
    assert choose_algorithm(scores,'2025-03','Baseline',Settings(min_improvement=4))[0]=='Baseline'
    incomplete=scores[~((scores.target_month=='2025-02') & (scores.algorithm=='XGBoost'))]
    assert choose_algorithm(incomplete,'2025-03','Baseline',Settings())[0]=='Baseline'


@pytest.mark.parametrize('task', ['ot','gate'])
def test_tournament_all_algorithms_future_prediction_and_no_leakage(panels, task):
    panel,gates=panels
    data=ot_examples(panel) if task=='ot' else gates
    config=Settings(min_train_rows=10)
    before,scores,choices=risk_tournament(data,task,config)
    assert set(before.algorithm)==set(ALGORITHMS)
    assert before.target_month.max()=='2026-01'
    assert before.loc[before.target_month=='2026-01','actual'].isna().all()
    assert (before.dropna(subset=['training_through']).training_through < before.dropna(subset=['training_through']).target_month).all()
    assert (choices.dropna(subset=['selection_through']).selection_through < choices.dropna(subset=['selection_through']).target_month).all()
    changed=data.copy()
    affected=(changed.target_month>='2025-09') & changed.actual.notna()
    changed.loc[affected,'actual'] = (1-changed.loc[affected,'actual']) if task=='gate' else changed.loc[affected,'actual']+100
    after,_,new_choices=risk_tournament(changed,task,config)
    cols=['entity','algorithm','prediction','selected']
    pd.testing.assert_frame_equal(before[before.target_month=='2025-09'][cols].reset_index(drop=True),after[after.target_month=='2025-09'][cols].reset_index(drop=True))
    pd.testing.assert_frame_equal(choices[choices.target_month=='2025-09'].reset_index(drop=True),new_choices[new_choices.target_month=='2025-09'].reset_index(drop=True))
    valid=scores[scores.status=='ok']
    assert valid.groupby(['development_type','target_month']).n.nunique().max()==1


def test_gate_single_class_abstains_learners(panels):
    data=panels[1].copy()
    data.loc[data.actual.notna(),'actual']=0
    pred,scores,choices=risk_tournament(data,'gate',Settings(min_train_rows=10))
    assert set(pred.algorithm)=={'Baseline'}
    assert scores.loc[scores.algorithm!='Baseline','status'].str.contains('부족|두 종류').all()
