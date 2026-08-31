"""Independent output and hard-constraint checks for the first-round artifacts."""
import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]
s=pd.read_csv(root/'results/result_2_group_schedule.csv')
v=pd.read_excel(next((root/'input').rglob('*.xlsx')),sheet_name='venues').set_index('venue_id')
slots=pd.read_excel(next((root/'input').rglob('*.xlsx')),sheet_name='time_slots')
times=pd.to_datetime(s.utc_datetime,utc=True)
rest=[]
for team in set(s.team_a)|set(s.team_b):
    x=times[(s.team_a==team)|(s.team_b==team)].sort_values(); rest.extend(x.diff().dropna().dt.total_seconds()/3600)
checks={'rows_72':len(s)==72,'venue_slot_unique':not s.duplicated(['venue_id','utc_datetime']).any(),'min_rest_hours':float(min(rest)),'rest_ge_60':min(rest)>=60,'venue_total_min':all(s.venue_id.value_counts().get(i,0)>=v.loc[i,'min_total_matches'] for i in v.index),'venue_total_max':all(s.venue_id.value_counts().get(i,0)<=v.loc[i,'max_total_matches'] for i in v.index)}
print(json.dumps(checks,ensure_ascii=False,indent=2))
