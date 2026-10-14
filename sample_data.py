"""Reproducible synthetic monthly employee/project assignments."""
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ['사번', '이름', '기준월', '팀', '개발Project명', '업무비중(%)', '개발Gate 단계',
           '개발Project난이도', '직무역량Level평균', 'Gate 완료일', '총OT(초과근로)',
           '중앙성(network분석결과)', '메일 발신량', '메일 수신량']


def generate_sample(year=2026, seed=42):
    rng = np.random.default_rng(seed)
    difficulty = rng.integers(1, 6, 30)
    capability = rng.uniform(2, 5, 500)
    assignments = [rng.choice(30, 3, replace=False) for _ in range(500)]
    rows = []
    for month in range(2, 10):
        period = f'{year}-{month:02d}'
        for i in range(500):
            ot = round(float(np.clip(rng.normal(18 + 3 * (month - 2), 12), 0, 80)), 2)
            centrality = round(float(rng.beta(2, 5)), 4)
            sent, received = int(rng.poisson(100)), int(rng.poisson(160))
            for project, share in zip(assignments[i], [30, 40, 30]):
                gate = min(5, 1 + (month - 2) // 2)
                # Only dates already observed by this month are recorded.
                done = f'{period}-20' if month % 2 == 1 else ''
                rows.append([f'E{i+1:04d}', f'직원{i+1:03d}', period, f'팀{i%20+1:02d}',
                             f'Project{project+1:02d}', share, gate, difficulty[project],
                             round(float(capability[i]), 2), done, ot, centrality, sent, received])
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == '__main__':
    path = Path(__file__).parent / 'data' / 'sample.csv'
    path.parent.mkdir(exist_ok=True)
    frame = generate_sample()
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'{path}: {len(frame):,} rows, 500 employees, 20 teams, 30 projects')
