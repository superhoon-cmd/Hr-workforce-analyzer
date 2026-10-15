import pandas as pd
from pathlib import Path
import streamlit as st
from sample_data import generate_sample
from analysis import METHODS, validate_csv, aggregate_projects, diagnose_projects, walk_forward, project_comparison, team_summary

st.set_page_config(page_title='프로젝트 인력 불균형 분석', layout='wide')
st.page_link('pages/2_team_models.py', label='새 팀별 후보모델 1·2·3 및 자동 알고리즘 선택')
st.title('프로젝트 인력 불균형 분석')
st.caption('CSV 업로드 → 관측 업무부담·유사 프로젝트 비교 → 다음 달 OT 예측')
st.warning('적정인원 정답을 가정하거나 필요·부족 인원 수를 산출하지 않습니다. 표시되는 가능성은 상대적인 점검 신호입니다. 여유 가능성은 납기·품질·산출물 확인이 필요합니다.')
sample = generate_sample()
st.download_button('샘플 CSV 다운로드 (2026년 2~9월 · 500명)', sample.to_csv(index=False).encode('utf-8-sig'), 'sample.csv', 'text/csv')
uploaded = st.file_uploader('CSV 업로드 (UTF-8 / UTF-8 BOM)', type=['csv'])
use_sample = st.checkbox('샘플 데이터로 분석', value=True)
with st.sidebar:
    st.header('점검 기준')
    high_ot = st.number_input('월 OT 고부담 기준(시간/FTE)', min_value=10.0, value=20.0)
    low_ot = st.number_input('월 OT 낮은 부담 기준(시간/FTE)', min_value=0.0, max_value=high_ot, value=10.0)
    min_peers = st.number_input('최소 유사 프로젝트 수', min_value=3, max_value=20, value=3)
    min_improvement = st.number_input('모델 교체 최소 MAE 개선 폭(시간/FTE)', min_value=0.0, value=0.0, step=0.1)
    st.caption('기준은 검토용 가정이며, 조정해도 ML 학습 정답은 관측 OT로 유지됩니다.')
if uploaded is None and not use_sample:
    st.info('CSV를 업로드하거나 샘플 데이터 분석을 선택하세요.')
    st.stop()


@st.cache_data(show_spinner='관측 신호 분석 및 월별 OT 학습 중…')
def analyze(frame, min_peers, high_ot, low_ot, min_improvement):
    valid = validate_csv(frame)
    projects = aggregate_projects(valid)
    diagnosis = diagnose_projects(projects, min_peers, high_ot, low_ot)
    predictions, metrics = walk_forward(projects, min_improvement,
                                        model_dir=Path(__file__).resolve().parent / '.cache' / 'models')
    return valid, projects, diagnosis, predictions, metrics


try:
    raw = pd.read_csv(uploaded, dtype={'사번': str}, encoding='utf-8-sig') if uploaded else sample
    valid, projects, diagnosis, predictions, metrics = analyze(raw, min_peers, high_ot, low_ot, min_improvement)
except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
    st.error(f'데이터를 확인하세요: {exc}')
    st.stop()

st.info('ML 후보는 데이터 시작부터 예측월 직전까지 확보된 전체 이력으로 누적 학습합니다. 기존 모델과 후보를 같은 월의 실제 OT로 비교하고, MAE가 개선될 때만 다음 달부터 교체합니다. 개선되지 않으면 기존 모델을 유지합니다. 최근 3개월 기준은 관측 신호의 추세 점검에만 사용합니다.')
month = st.selectbox('관측·예측 평가 월', sorted(predictions['기준월'].unique()))
current = diagnosis[diagnosis['기준월'] == month].copy()
comparison = project_comparison(predictions, month)
teams = team_summary(valid, diagnosis, month)
columns = st.columns(4)
columns[0].metric('관측 투입 FTE', f"{current.current_fte.sum():.1f}")
for column, status in zip(columns[1:], ['부족 가능성', '여유 가능성', '판단 유보']):
    column.metric(f'{status} 프로젝트', str(current['판단'].eq(status).sum()))

signal_tab, forecast_tab, team_tab, validation_tab, data_tab = st.tabs(['관측 인력 신호', 'OT 예측 비교', '팀별 부담', '월별 검증', '데이터'])
with signal_tab:
    st.subheader('프로젝트별 관측 부담과 판단 근거')
    display = current.rename(columns={'current_fte': '투입 FTE', 'ot_per_fte': '평균 OT(시간/FTE)', 'previous_avg_ot': '전달 평균 OT(시간/FTE)', 'ot_change': '전달 대비 OT 변화(시간/FTE)', 'max_contribution_share': '최대 참여자 기여 비율', 'contribution_hhi': '기여 집중도(HHI)'})
    cols = ['개발Project명', '판단', '근거', '투입 FTE', '평균 OT(시간/FTE)', '전달 평균 OT(시간/FTE)', '전달 대비 OT 변화(시간/FTE)', '유사 프로젝트 수', '유사 평균 OT 중앙값(시간/FTE)', '유사 투입 FTE 중앙값', '3개월 OT 고부담', '3개월 OT 상승', '최대 참여자 기여 비율', '기여 집중도(HHI)']
    st.dataframe(display[cols], hide_index=True, width='stretch')
    st.bar_chart(display.set_index('개발Project명')[['평균 OT(시간/FTE)', '유사 평균 OT 중앙값(시간/FTE)']])
    st.download_button('프로젝트 점검 신호 CSV', display[cols].to_csv(index=False).encode('utf-8-sig'), f'project-signals-{month}.csv', 'text/csv')
with forecast_tab:
    st.subheader('직전월 정보로 예측한 해당 월 OT 비교')
    st.caption('단위: 평균 OT 시간/FTE. 이전월 유지가 기준 모델이며 모델 차이는 신뢰구간이 아닙니다. 신규·중단 프로젝트는 연속 월 평가에서 제외될 수 있습니다.')
    st.bar_chart(comparison.set_index('개발Project명')[[*METHODS, '관측 OT(시간/FTE)']])
    st.dataframe(comparison, hide_index=True, width='stretch')
    st.write(f'평가 포함 {len(comparison)} / 관측 프로젝트 {len(current)}개')
    excluded = current.loc[~current['개발Project명'].isin(comparison['개발Project명']), '개발Project명']
    if len(excluded):
        st.info('직전월 자료가 없어 예측에서 제외: ' + ', '.join(excluded))
    st.download_button('OT 예측 비교 CSV', comparison.to_csv(index=False).encode('utf-8-sig'), f'ot-forecast-{month}.csv', 'text/csv')
with team_tab:
    st.caption('같은 월의 관측 프로젝트 판단에 대한 팀의 기여 비중입니다. 부족 가능성 프로젝트 기여 비중이 높아도 해당 팀의 부족 인원 수를 뜻하지 않습니다.')
    st.dataframe(teams, hide_index=True, width='stretch')
    st.bar_chart(teams.set_index('팀')[['평균 OT(시간/FTE)']])
    st.download_button('팀별 부담 CSV', teams.to_csv(index=False).encode('utf-8-sig'), f'team-workload-{month}.csv', 'text/csv')
with validation_tab:
    st.caption('정답은 CSV에서 관측된 다음 달 OT입니다. MAE/RMSE 단위는 시간/FTE이며 인원 정답이나 가정식 FTE를 학습하지 않습니다. 합성 데이터 성능은 실데이터 정확도를 입증하지 않습니다.')
    st.dataframe(metrics, hide_index=True, width='stretch')
    st.caption('학습 건수는 해당 월 예측에 실제 사용한 고정 모델의 학습량입니다. 누적 후보 학습 건수는 새 후보의 전체 학습량입니다. 평가월의 실제 OT는 다음 달 모델 선택에만 사용합니다. 최종 선택 모델과 학습 범위는 서버에 저장하며, CSV 변경 시 전체 이력으로 선택 과정을 재현합니다.')
    st.download_button('월별 OT 평가 CSV', metrics.to_csv(index=False).encode('utf-8-sig'), 'ot-metrics.csv', 'text/csv')
    st.dataframe(projects.groupby('기준월')['개발Project명'].nunique().rename('관측 프로젝트 수'))
with data_tab:
    st.write(f"{len(valid):,}행 · {valid['사번'].nunique()}명 · {valid['팀'].nunique()}팀 · {valid['개발Project명'].nunique()}프로젝트")
    st.dataframe(valid.head(100), hide_index=True, width='stretch')
    st.caption('업로드 데이터는 외부 API로 전송하지 않으며 서버 메모리 캐시에 보관됩니다.')
with st.expander('판단 기준과 한계'):
    st.markdown('''투입 FTE = Σ(업무비중/100). 배분 OT = Σ(직원 월 OT × 업무비중/100).
    평균 OT(시간/FTE) = 배분 OT / 투입 FTE. 실제 근로시간이 아니라 OT 부담 지표입니다.

    유사 프로젝트는 같은 월의 다른 프로젝트 중 난이도 차이 ≤1, Gate 차이 ≤1, 역량 차이 ≤0.75입니다.
    업무규모가 같다는 뜻은 아닙니다.
    부족 가능성: 투입 FTE ≤ 유사 중앙값의 90%, OT가 고부담 기준 이상이고 유사 OT의 125% 이상 및
    유사 OT보다 5시간 이상 높음, 최근 연속 3개월 고부담 또는 매월 2시간 초과 상승.
    여유 가능성: 투입 FTE ≥ 유사 중앙값의 110%, OT가 낮은 부담 기준 이하이고 유사 OT의 75% 이하
    (유사 OT 중앙값 >0). 나머지는 판단 유보이며 정상 인원 판정이 아닙니다.

    전달 평균 OT는 자동 계산하며 미관측이면 빈 값입니다. ML에는 미관측 여부와 함께 입력합니다.
    메일량·중앙성은 생산성 정답이 아니며 Gate 완료일만으로 납기 준수를 알 수 없습니다.
    계획·실적 공수, 목표 완료일, 산출물, 품질·재작업 자료가 있어야 적정인원 범위를 검증할 수 있습니다.
    인원 변경의 인과 효과나 채용·감축 필요 인원은 산출하지 않습니다.''')
