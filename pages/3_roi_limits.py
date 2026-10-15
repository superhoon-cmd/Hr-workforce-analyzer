"""Manual or Excel-based project FTE limit monitoring."""
from pathlib import Path
import sys
from hashlib import sha256
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from roi_limits import COLUMNS, LABELS, read_input, assess_limits

st.set_page_config(page_title='개발조직 FTE 한도', layout='wide')
st.title('개발조직 FTE 한도')
st.markdown('프로젝트별 **개발조직의 참여율 합산 FTE**를 직접 입력하거나 엑셀로 업로드하세요. '
            '여러 개발팀이 같은 프로젝트에 참여하면 합산하고, 개발조직 외 시험·품질·PM·지원조직은 제외합니다. '
            '직무명이 아닌 소속 조직을 기준으로 합니다. OT를 FTE에 추가하지 않습니다.')
st.caption('인사·WBS 원자료 없이 사용할 수 있습니다. 입력값은 예측모델의 학습자료를 변경하지 않습니다.')
left, right = st.columns(2)
cell_limit = left.number_input('Cell 기본 한도(FTE)', min_value=0., value=11., step=.5)
pack_limit = right.number_input('Pack 기본 한도(FTE)', min_value=0., value=14., step=.5)
st.caption('11·14는 대화에서 제시한 초기값입니다. 적용할 기준으로 수정하세요. 프로젝트 한도가 입력된 행은 해당 값을 우선 사용하며, 공란이면 기본값을 적용합니다. 0은 실제 한도 0으로 처리합니다.')
with st.expander('엑셀 입력 형식'):
    st.write('첫 시트의 첫 행에 아래 열 이름을 넣으세요. 기준월과 프로젝트 ID는 텍스트로 저장하고, FTE는 소수 숫자로 입력하세요. 수식은 값으로 붙여넣은 후 업로드하세요.')
    st.code('기준월 | 프로젝트 ID | 개발 구분 | 개발조직 투입 FTE | 프로젝트 한도 FTE')
    st.write('예: 2026-10 / CELL-A / Cell / 12.5 / 11 → 1.5 FTE 초과. 프로젝트 한도 FTE 열은 생략할 수 있습니다.')
    template = pd.DataFrame(columns=list(LABELS.values()))
    st.download_button('엑셀용 빈 CSV 양식', template.to_csv(index=False).encode('utf-8-sig'), 'development_fte_template.csv', 'text/csv')
upload = st.file_uploader('엑셀(.xlsx) 또는 CSV 업로드', type=['xlsx', 'csv'])
source = pd.DataFrame({c: pd.Series(dtype='float64' if c.endswith('_fte') else 'str') for c in COLUMNS})
editor_key = 'roi_manual'
if upload is not None:
    try:
        raw = upload.getvalue()
        source = read_input(raw, upload.name)
        # Validate file content before allowing editing; do not silently drop duplicate rows.
        source = assess_limits(source, cell_limit, pack_limit).reindex(columns=COLUMNS)
        for col in ['development_fte', 'limit_fte']:
            source[col] = pd.to_numeric(source[col], errors='coerce')
        editor_key = 'roi_upload_' + sha256(raw).hexdigest()
    except Exception as exc:
        st.error(f'업로드 파일을 확인하세요: {exc}')
        st.stop()
st.caption('행을 추가하여 직접 입력하거나 업로드한 값을 수정할 수 있습니다. 투입 FTE 공란은 0으로 간주하지 않습니다.')
edited = st.data_editor(source, num_rows='dynamic', hide_index=True, key=editor_key,
    column_config={
        'month': st.column_config.TextColumn(LABELS['month'], help='YYYY-MM'),
        'project_id': st.column_config.TextColumn(LABELS['project_id']),
        'family': st.column_config.SelectboxColumn(LABELS['family'], options=['Cell', 'Pack']),
        'development_fte': st.column_config.NumberColumn(LABELS['development_fte'], min_value=0., step=.1),
        'limit_fte': st.column_config.NumberColumn(LABELS['limit_fte'], min_value=0., step=.1),
    })
confirmed = st.checkbox('입력한 투입 FTE에는 개발조직 소속 인력만 포함되어 있습니다.', key='roi_scope_' + editor_key)
if not confirmed:
    st.info('집계 대상 조직을 확인하면 한도 비교 결과가 표시됩니다. 합산값만으로 소속 조직을 자동 검증할 수는 없습니다.')
    st.stop()
try:
    result = assess_limits(edited, cell_limit, pack_limit)
except ValueError as exc:
    st.error(str(exc)); st.stop()
if result.empty:
    st.info('직접 행을 추가하거나 파일을 업로드하세요.'); st.stop()
month = st.selectbox('한도 확인월', sorted(result.month.unique()), index=len(result.month.unique())-1)
current = result[result.month == month]
st.write({'한도 초과 건수': int(current.status.eq('한도 초과').sum()), '투입 FTE 미입력 건수': int(current.development_fte.isna().sum())})
display_labels = {**LABELS, 'scope':'포함 범위', 'limit_source':'한도 출처', 'effective_limit_fte':'적용 한도 FTE',
                  'difference_fte':'투입−한도 FTE', 'excess_fte':'초과 FTE', 'remaining_fte':'잔여 FTE', 'status':'상태'}
st.dataframe(current.rename(columns=display_labels), hide_index=True)
st.download_button('입력값 보존·재업로드용 CSV', edited.rename(columns=LABELS).to_csv(index=False).encode('utf-8-sig'), 'development_fte_input.csv', 'text/csv')
st.download_button('전체 월 한도 비교 결과 CSV', result.rename(columns=display_labels).to_csv(index=False).encode('utf-8-sig'), 'development_fte_limits.csv', 'text/csv')
st.caption('입력은 현재 세션에서 사용합니다. 종료 전에 CSV를 내려받아 보존하세요. 초과량은 감축 권고나 적정인원 추정치가 아니며, 잔여 한도는 추가 인력이 필요하다는 의미가 아닙니다.')
