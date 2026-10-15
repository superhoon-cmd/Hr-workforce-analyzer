"""Team models V2; retain the existing project dashboard as a separate page."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from workforce_data import generate_snapshots, build_panels
from workforce_models import Settings, normalize_ot, peer_backtest, ot_examples, risk_tournament

st.set_page_config(page_title='팀별 적정인원 후보모델', layout='wide')
st.title('팀별 적정인원 후보모델')
st.caption('모델 1: 시간 환산 · 모델 2: 유사사례 + 분위수 회귀 · 모델 3: 네 알고리즘 자동 비교')
st.warning('인원범위는 조건부 참고치입니다. 품질·실제 업무범위와 증원 효과는 검증하지 않았으며, 감축 정원이나 인력 변경의 인과효과를 산출하지 않습니다.')
with st.sidebar:
    st.subheader('운영 가정')
    target_ot = st.number_input('모델 1 기준 OT(시간/인)', min_value=0., value=10.)
    leave_reference = st.number_input('계획휴가 미확보 시 정상휴가 가정(시간/인·월)', min_value=0., value=8.)
    stable_ot = st.number_input('안정사례 개인 OT P90 상한', min_value=0., value=20.)
    min_peers = st.number_input('최소 비교사례 수', min_value=10, value=20)
    min_teams = st.number_input('최소 비교 팀 수', min_value=2, value=3)
    min_train = st.number_input('모델 3 최소 학습 건수', min_value=10, value=20)
    window = st.number_input('모델 선택 평가창(최근 공통 월)', min_value=2, value=3)
    st.caption('최소 공통 검증 2개월 이후 선택. OT: MAE, Gate: Brier. 미세한 성능 차이도 기본적으로 반영합니다.')
    ot_margin = st.number_input('OT 교체 최소 MAE 개선', min_value=0., value=0., step=.1)
    gate_margin = st.number_input('Gate 교체 최소 Brier 개선', min_value=0., value=0., step=.001, format='%.3f')

demo = st.checkbox('합성 샘플로 분석', value=True)
sample = generate_snapshots()
with st.expander('입력 CSV 3종과 샘플 다운로드', expanded=not demo):
    st.markdown('직원·월 / 직원·프로젝트·월 / 프로젝트·Gate·월 자료를 별도로 올리세요. '
                '`normal_hours`는 **휴가를 포함한 월 소정시간**(월중 재직기간 반영), '
                '`share`는 0~1 비율입니다. 조직명에는 Cell 또는 Pack이 필요합니다. '
                '`planned_leave_hours`는 해당 월 시작 전에 기록한 계획 휴가입니다. '
                'Gate 원래 예정일을 유지하고 변경 예정일과 완료일은 당시 스냅샷으로 보존하세요. '
                '기존 프로젝트 CSV는 기존 화면에서 사용할 수 있습니다.')
    uploads = []
    for label, frame in zip(['employees', 'assignments', 'gates'], sample):
        st.download_button(f'{label} 합성 CSV', frame.to_csv(index=False).encode('utf-8-sig'), f'{label}.csv', 'text/csv')
        uploads.append(st.file_uploader(f'{label}.csv', type='csv', key=label))

if any(u is not None for u in uploads):
    if not all(u is not None for u in uploads):
        st.info('업로드 분석에는 CSV 3종이 모두 필요합니다. 부분 자료를 샘플과 혼합하지 않습니다.')
        st.stop()
    try:
        frames = [pd.read_csv(u, dtype={'employee_id':str, 'project_id':str, 'gate_id':str}, encoding='utf-8-sig') for u in uploads]
    except (ValueError, UnicodeError) as exc:
        st.error(str(exc)); st.stop()
elif demo:
    frames = sample
    st.info('현재 자료는 18개월 합성 예시입니다. 표시 성능은 실제 회사 정확도를 의미하지 않습니다.')
else:
    st.info('CSV 3종을 올리거나 합성 샘플을 선택하세요.')
    st.stop()

if 'planned_leave_hours' not in frames[0].columns:
    frames[0] = frames[0].copy()
    frames[0]['planned_leave_hours'] = leave_reference
    st.warning(f'계획휴가 자료가 없어 정상휴가 {leave_reference:g}시간/인·월을 가정합니다. '
               '모델 1 환산과 모델 2 안정사례 선정에 이 가정이 적용됩니다. 실제 계획으로 검증된 값이 아닙니다.')


@st.cache_data(show_spinner='팀별 사례 추정과 네 알고리즘의 시간순 검증 중…')
def run_models(e, a, g, config, gate_config):
    panel, gate = build_panels(e, a, g)
    peer = peer_backtest(panel, config)
    ot_result = risk_tournament(ot_examples(panel), 'ot', config)
    gate_result = risk_tournament(gate, 'gate', gate_config)
    return panel, peer, ot_result, gate_result


config = Settings(stable_ot=stable_ot, min_peer_rows=int(min_peers), min_peer_teams=int(min_teams),
                  min_train_rows=int(min_train), selection_months=int(window), min_improvement=ot_margin)
gate_config = Settings(min_train_rows=int(min_train), selection_months=int(window), min_improvement=gate_margin)
try:
    panel, peer, ot_result, gate_result = run_models(*frames, config, gate_config)
except (ValueError, KeyError) as exc:
    st.error(f'자료를 확인하세요: {exc}')
    st.stop()

month = st.selectbox('분석 기준월', sorted(panel.month.unique()), index=len(panel.month.unique())-1)
target = str(pd.Period(month, 'M')+1)
tab1, tab2, tab3, tab4 = st.tabs(['모델 1 · 시간 환산', '모델 2 · 비교범위', '모델 3 · 자동 선택', '검증·다운로드'])
with tab1:
    st.caption('관측 정상 가용시간 + OT를 계획 가용시간 + 기준 OT로 환산합니다. 다음 달 업무수요 예측은 아닙니다.')
    st.dataframe(normalize_ot(panel[panel.month == month], target_ot), hide_index=True)
with tab2:
    st.markdown('**알고리즘: Cell/Pack별 표준화 + L1 정규화 선형 분위수 회귀(P10·P50·P90).** '
                '유사사례의 실제 인원 분위수도 함께 제공합니다. 현재 HC·FTE·가용시간·OT를 '
                '인원 회귀의 입력에서 제외합니다. 계층모형은 구현하지 않았습니다.')
    st.caption('이전 월의 일정·OT·휴가 기준 안정사례만 학습합니다. P10~P90은 경험적 분포범위이며 최소 정원 또는 80% 신뢰구간이 아닙니다. 분위수 교차는 정렬하고 표시합니다.')
    st.dataframe(peer[peer.month == month], hide_index=True)
with tab3:
    st.markdown('**OT:** 이전월 유지 / Ridge 회귀 / Random Forest / XGBoost → **MAE 최소화**  \n'
                '**Gate:** 과거 지연빈도 / Logistic 회귀 / Random Forest / XGBoost → **Brier 최소화**')
    st.caption('Cell/Pack·예측대상별 별도 선택. 예측월보다 앞선 공통 평가창만 사용하며, 모든 학습 가능 알고리즘을 매월 누적 재학습합니다. 기준모델이 더 좋으면 기준모델을 유지합니다.')
    for title, (pred, scores, choices) in [('OT(시간/인)', ot_result), ('Gate 지연확률', gate_result)]:
        st.subheader(f'{target} {title}')
        if choices.empty or choices[choices.target_month == target].empty:
            st.info('예측 대상 또는 비교할 Gate 기록이 없습니다.')
            continue
        st.dataframe(choices[choices.target_month == target], hide_index=True)
        if not pred.empty:
            selected = pred[(pred.target_month == target) & pred.selected]
            st.dataframe(selected, hide_index=True)
        with st.expander(f'{title} 전체 후보 결과'):
            if not pred.empty:
                st.dataframe(pred[pred.target_month == target], hide_index=True)
            st.dataframe(scores[scores.target_month == target], hide_index=True)
    st.caption('Gate는 프로젝트·Gate당 최초 평가기회 한 번만 사용합니다. 해당 기회는 당시 예정일이 다음 달에 들어올 때이며, 이후 일정 변경으로 과거 정답을 바꾸지 않습니다. 인력 임계점·증원 효과·퇴직원인은 산출하지 않습니다.')
with tab4:
    st.subheader('모델 2 시간순 검증')
    evaluated = peer[peer.evaluated]
    st.write({'평가 가능한 안정사례':len(evaluated),
              'P10~P90 포함률':float(evaluated.covered.mean()) if len(evaluated) else None,
              '평균 범위 폭':float(evaluated.interval_width.mean()) if len(evaluated) else None})
    st.caption('포함률은 이후에 안정운영으로 확인된 사례에 대한 조건부 평가이며, 최소인원 정확도가 아닙니다.')
    outputs = {'team_panel':panel, 'model2_backtest':peer}
    for name, result in [('ot', ot_result), ('gate', gate_result)]:
        for suffix, frame in zip(['predictions', 'scores', 'selections'], result):
            outputs[f'{name}_{suffix}'] = frame
        st.subheader(f'{name} 전체 기간 후보 평가')
        st.dataframe(result[1], hide_index=True)
        if not result[0].empty:
            selected = result[0][result[0].selected & result[0].actual.notna()]
            if len(selected):
                error = selected.prediction-selected.actual
                st.write({'선택된 예측의 실현 오차':float((error.abs() if name == 'ot' else error**2).mean()), '평가건수':len(selected)})
    for name, frame in outputs.items():
        st.download_button(f'{name}.csv 다운로드', frame.to_csv(index=False).encode('utf-8-sig'), f'{name}.csv', 'text/csv')
    st.caption('업로드마다 분석을 재현합니다. 자료는 서버 메모리 캐시에 보관되며 외부 API로 보내지 않습니다. 예측·선택 이력은 CSV로 보존하세요. 사용자 인증과 영구 예측 등록소는 별도 운영 구축이 필요합니다.')
