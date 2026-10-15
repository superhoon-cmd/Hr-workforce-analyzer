"""Development-only monthly project FTE limits, independent of model predictions."""
from io import BytesIO
import numpy as np
import pandas as pd

COLUMNS = ['month', 'project_id', 'family', 'development_fte', 'limit_fte']
LABELS = dict(zip(COLUMNS, ['기준월', '프로젝트 ID', '개발 구분', '개발조직 투입 FTE', '프로젝트 한도 FTE']))


def read_input(data: bytes, filename: str) -> pd.DataFrame:
    """Read literal values from the first Excel sheet or CSV; preserve text IDs."""
    if filename.lower().endswith('.xlsx'):
        frame = pd.read_excel(BytesIO(data), engine='openpyxl', dtype=str)
    elif filename.lower().endswith('.csv'):
        frame = pd.read_csv(BytesIO(data), encoding='utf-8-sig', dtype=str)
    else:
        raise ValueError('xlsx 또는 UTF-8 CSV 파일을 사용하세요.')
    frame = frame.rename(columns={v: k for k, v in LABELS.items()})
    if frame.columns.duplicated().any():
        raise ValueError('같은 의미의 열이 중복되었습니다.')
    return frame


def assess_limits(frame: pd.DataFrame, cell_limit: float, pack_limit: float) -> pd.DataFrame:
    for value in (cell_limit, pack_limit):
        if not np.isfinite(value) or value < 0:
            raise ValueError('기본 한도는 0 이상의 유한한 숫자여야 합니다.')
    required = COLUMNS[:-1]
    if missing := set(required) - set(frame.columns):
        raise ValueError(f'필수 열 누락: {sorted(missing)}')
    out = frame.copy().map(lambda x: np.nan if isinstance(x, str) and not x.strip() else x).dropna(how='all')
    if 'limit_fte' not in out:
        out['limit_fte'] = np.nan
    for key in ['month', 'project_id', 'family']:
        if out[key].isna().any():
            raise ValueError(f'{LABELS[key]}을 입력하세요.')
        out[key] = out[key].astype(str).str.strip()
    if not out.month.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():
        raise ValueError('기준월은 YYYY-MM 텍스트로 입력하세요. 예: 2026-10')
    out['family'] = out.family.str.lower().map({'cell': 'Cell', 'pack': 'Pack'})
    if out.family.isna().any():
        raise ValueError('개발 구분은 Cell 또는 Pack이어야 합니다.')
    if out.duplicated(['month', 'project_id', 'family']).any():
        raise ValueError('기준월·프로젝트·개발 구분별로 한 행만 입력하세요. 여러 개발팀의 FTE는 먼저 합산하세요.')
    for key in ['development_fte', 'limit_fte']:
        values = pd.to_numeric(out[key], errors='coerce')
        bad = out[key].notna() & (values.isna() | ~np.isfinite(values) | values.lt(0))
        if bad.any():
            raise ValueError(f'{LABELS[key]}는 0 이상의 숫자 또는 공란이어야 합니다.')
        out[key] = values
    out['scope'] = '개발조직만'
    out['limit_source'] = np.where(out.limit_fte.notna(), '프로젝트별 입력', '개발 구분 기본값')
    out['effective_limit_fte'] = out.limit_fte.fillna(out.family.map({'Cell': cell_limit, 'Pack': pack_limit}))
    out['difference_fte'] = out.development_fte - out.effective_limit_fte
    out['excess_fte'] = out.difference_fte.clip(lower=0)
    out['remaining_fte'] = (-out.difference_fte).clip(lower=0)
    out['status'] = np.select([out.development_fte.isna(), out.difference_fte > 1e-9], ['투입 FTE 미입력', '한도 초과'], default='한도 이내')
    return out.sort_values(['month', 'project_id', 'family']).reset_index(drop=True)
