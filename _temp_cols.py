import pandas as pd

s = pd.read_csv('subgnn_benchmark_results.csv')
j = pd.read_csv('arxiv_eval.csv')

print('=== SubGNN columns (%d) ===' % len(s.columns))
for c in s.columns:
    vals = s[c].dropna()
    sample = vals.iloc[0] if len(vals) > 0 else 'N/A'
    print('  %-40s dtype=%-10s sample=%s' % (c, str(s[c].dtype), sample))

print()
print('=== Jigsaw columns (%d) ===' % len(j.columns))
for c in j.columns:
    vals = j[c].dropna()
    sample = vals.iloc[0] if len(vals) > 0 else 'N/A'
    print('  %-40s dtype=%-10s sample=%s' % (c, str(j[c].dtype), sample))
