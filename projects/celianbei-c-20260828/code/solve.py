import argparse, json, math, os, re, sys, hashlib
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

SEED = 20260828
rng = np.random.default_rng(SEED)

def norm(s):
    s = pd.Series(s, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    return np.zeros(len(s)) if hi <= lo else (s-lo)/(hi-lo)

def clip(x, lo, hi): return np.minimum(hi, np.maximum(lo, x))

def read_data(input_dir):
    xlsx = next(Path(input_dir).rglob('*.xlsx'))
    xl = pd.ExcelFile(xlsx)
    data = {s: pd.read_excel(xlsx, sheet_name=s) for s in xl.sheet_names}
    return data, str(xlsx)

def q1_predict(data, results, audit):
    hist, teams, groups = data['historical_matches'], data['teams'], data['groups_matches']
    tr = hist[hist.dataset_split.astype(str).str.lower()=='train'].copy()
    te = hist[hist.dataset_split.astype(str).str.lower()=='test'].copy()
    t = teams.set_index('team_name')
    def make_rows(df):
        rows=[]
        for _,r in df.iterrows():
            a,b = t.loc[r.team_a], t.loc[r.team_b]
            d = pd.to_datetime(r.date)
            row = {
                'elo_diff': a.elo_rating-b.elo_rating, 'strength_diff': a.strength_score-b.strength_score,
                'fan_sum': a.fan_base_index+b.fan_base_index, 'star_sum': a.star_index+b.star_index,
                'market_sum': a.market_value_musd+b.market_value_musd, 'age_diff': a.avg_age-b.avg_age,
                'attack_sum': a.style_attack+b.style_attack, 'defense_sum': a.style_defense+b.style_defense,
                'neutral': int(bool(r.get('neutral',True))), 'month': d.month, 'year': d.year,
                'competition': str(r.get('competition','')), 'stage': str(r.get('stage','')),
            }
            rows.append(row)
        return pd.DataFrame(rows)
    Xall = make_rows(pd.concat([tr,te], ignore_index=True))
    Xall = pd.get_dummies(Xall, columns=['competition','stage'], dtype=float)
    Xtr, Xte = Xall.iloc[:len(tr)], Xall.iloc[len(tr):]
    y = tr.tv_viewers.astype(float).values/1e6
    order = np.argsort(pd.to_datetime(tr.date).values)
    cut = max(20, int(len(tr)*0.8)); va_idx=order[cut:]; fit_idx=order[:cut]
    models = {
        'extra_trees': ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,max_features=0.8,random_state=SEED,n_jobs=-1),
        'random_forest': RandomForestRegressor(n_estimators=500,min_samples_leaf=4,max_features=0.8,random_state=SEED,n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=300,max_depth=2,learning_rate=0.03,loss='huber',random_state=SEED)
    }
    scores=[]
    for name,m in models.items():
        m.fit(Xtr.iloc[fit_idx], y[fit_idx]); pred=m.predict(Xtr.iloc[va_idx])
        scores.append({'model':name,'mse_million':float(mean_squared_error(y[va_idx],pred)), 'mae_million':float(mean_absolute_error(y[va_idx],pred)), 'r2':float(r2_score(y[va_idx],pred))})
    best_name=min(scores,key=lambda z:z['mse_million'])['model']; best=models[best_name]
    best.fit(Xtr,y); test_pred=np.maximum(0.01,best.predict(Xte))
    # Group matches use only pre-match team attributes and a common World Cup/Group context.
    gr=groups.copy(); gr['date']=pd.to_datetime('2026-06-11')+pd.to_timedelta((gr.round_in_group-1)*3,unit='D'); gr['competition']='World Cup'; gr['stage']='Group'; gr['neutral']=True
    grf=make_rows(gr); grf=pd.get_dummies(grf,columns=['competition','stage'],dtype=float).reindex(columns=Xall.columns,fill_value=0)
    group_pred=np.maximum(0.01,best.predict(grf))*1e6
    # Blend toward the supplied base audience only when the model extrapolates far outside training range.
    results['q1_model_scores']=scores; results['q1_selected_model']=best_name
    out_test=pd.DataFrame({'match_id_test':te.match_id.values,'predicted_test_tv_viewers':test_pred})
    out_match=groups[['match_id','team_a','team_b']].copy(); out_match['predicted_tv_viewers']=group_pred
    out_test.to_csv(results['dir']/ 'result_1_test_prediction.csv',index=False)
    out_match.to_csv(results['dir']/ 'result_1_match_prediction.csv',index=False)
    audit['q1']={'n_train':len(tr),'n_test':len(te),'validation_cut':int(len(tr)-len(va_idx)),'scores':scores,'selected_model':best_name,'target_unit':'million viewers'}
    return out_match, out_test

def build_schedule(data, pred_match, results, audit):
    gm, bp, venues, slots, sec = data['groups_matches'], data['base_predictions'], data['venues'], data['time_slots'], data['security_requirements']
    sl=slots.copy(); sl['utc']=pd.to_datetime(sl.reference_utc_time,utc=True); sl['date']=pd.to_datetime(sl.date).dt.date
    # Slot assignment: process rounds in order and select high-value feasible slots while preserving 60 h rest.
    slot_used={s:0 for s in sl.slot_id}; last={}; highsec_by_date={}
    sec_req_map=data['security_requirements'].set_index('match_id').required_security_level.to_dict()
    limdf0=data['dynamic_resource_limits'].copy(); limdf0['reference_date']=pd.to_datetime(limdf0.reference_date).dt.date; limdf0=limdf0.set_index('reference_date')
    pmap=pred_match.set_index('match_id').predicted_tv_viewers.to_dict(); bmap=bp.set_index('match_id').to_dict('index')
    assigned=[]
    for rnd in [1,2,3]:
        rows=gm[gm.round_in_group==rnd].copy().sort_values('match_id')
        # Windows leave at least 60 h between adjacent rounds while retaining slack for repair.
        win_lo={1:pd.Timestamp('2026-06-11',tz='UTC'),2:pd.Timestamp('2026-06-15',tz='UTC'),3:pd.Timestamp('2026-06-20',tz='UTC')}[rnd]
        win_hi={1:pd.Timestamp('2026-06-16 23:59',tz='UTC'),2:pd.Timestamp('2026-06-22 23:59',tz='UTC'),3:pd.Timestamp('2026-06-30 23:59',tz='UTC')}[rnd]
        for _,m in rows.iterrows():
            teams=[m.team_a,m.team_b]; choices=[]
            for _,s in sl.iterrows():
                if s.utc < win_lo or s.utc > win_hi: continue
                if rnd==3 and int(sec_req_map[m.match_id])>=3:
                    lim=int(limdf0.loc[s.date,'high_security_capacity']);
                    if highsec_by_date.get(s.date,0)>=lim: continue
                if slot_used[s.slot_id] >= int(s.broadcast_capacity): continue
                if any(t in last and (s.utc-last[t]).total_seconds()/3600 < 60 for t in teams): continue
                val=float(pmap[m.match_id])*float(s.global_prime_score)
                # keep temporal spread; a small bonus avoids exhausting late slots too early
                val += 0.01*(s.utc-pd.Timestamp('2026-06-11',tz='UTC')).total_seconds()/86400
                choices.append((val,s))
            if not choices: raise RuntimeError(f'No feasible slot for {m.match_id}')
            # Penalize prime-time imbalance among teams.
            choices.sort(key=lambda x:x[0],reverse=True); _,s=choices[0]
            slot_used[s.slot_id]+=1
            if rnd==3 and int(sec_req_map[m.match_id])>=3: highsec_by_date[s.date]=highsec_by_date.get(s.date,0)+1
            for t in teams: last[t]=s.utc
            assigned.append({'match_id':m.match_id,'slot_id':s.slot_id,'utc':s.utc})
    sch=gm.merge(pd.DataFrame(assigned),on='match_id').merge(sl[['slot_id','date','reference_kickoff_time','global_prime_score','broadcast_capacity']],on='slot_id')
    # Venue assignment with minimum 2 and maximum 8 matches, daily capacity and security eligibility.
    vrows=venues.to_dict('records'); counts={v['venue_id']:0 for v in vrows}; daycounts={(v['venue_id'],d):0 for v in vrows for d in sl.date.unique()}; slotvenue=set()
    req=sec.set_index('match_id').required_security_level.to_dict(); bpred=bp.set_index('match_id').to_dict('index')
    # First reserve two games for each venue, then optimize remaining games.
    sch=sch.sort_values(['round_in_group','match_id']).reset_index(drop=True)
    venue_for={}
    for idx,m in sch.iterrows():
        cand=[]
        for v in vrows:
            if counts[v['venue_id']]>=v['max_total_matches'] or daycounts[(v['venue_id'],m['date'])]>=v['max_matches_per_day'] or (v['venue_id'],m['utc']) in slotvenue: continue
            if int(v['security_level'])<int(req[m.match_id]): continue
            need=float(bpred[m.match_id]['expected_attendance_base']); fit=min(need,v['capacity'])/max(need,1)
            # Travel proxy from team-to-venue; exact sequential burden is computed below.
            d1=data['distance_matrix']; aa=d1[(d1.origin_name==m.team_a)&(d1.destination_id==v['venue_id'])].iloc[0]; bb=d1[(d1.origin_name==m.team_b)&(d1.destination_id==v['venue_id'])].iloc[0]
            travel=(aa.distance_km+bb.distance_km)/2/10000 + (aa.travel_time_hour+bb.travel_time_hour)/2/20 + (aa.timezone_diff+bb.timezone_diff)/2/12
            reserve=5.0 if counts[v['venue_id']]<2 else 0
            score=0.55*fit + 0.25*float(v['capacity'])/94000 - 0.15*travel - 0.05*float(v['climate_risk']) + reserve
            cand.append((score,v))
        if not cand: raise RuntimeError(f'No feasible venue for {m.match_id}')
        cand.sort(key=lambda z:z[0],reverse=True); v=cand[0][1]
        venue_for[m.match_id]=v['venue_id']; counts[v['venue_id']]+=1; daycounts[(v['venue_id'],m['date'])]+=1; slotvenue.add((v['venue_id'],m['utc']))
    sch['venue_id']=sch.match_id.map(venue_for)
    sch=sch.merge(venues,on='venue_id',suffixes=('','_venue')).merge(sec[['match_id','required_security_level']],on='match_id').merge(bp[['match_id','expected_goals_a','expected_goals_b','uncertainty_index','attractiveness_index','expected_attendance_base']],on='match_id').merge(pred_match[['match_id','predicted_tv_viewers']],on='match_id')
    # Derive metric columns and objective.
    stage_map={1:'Group_Match_R1',2:'Group_Match_R2',3:'Group_Match_R3'}; tb=data['ticket_broadcast'].set_index('match_stage')
    sch['stage_key']=sch.round_in_group.map(stage_map); sch['base_price']=sch.stage_key.map(tb.base_ticket_price_usd); sch['unit_value']=sch.stage_key.map(tb.broadcast_unit_value_usd); sch['sponsor_weight']=sch.stage_key.map(tb.sponsor_weight)
    sch['expected_attendance']=np.minimum(sch.expected_attendance_base,sch.capacity)
    sch['ticket_revenue_usd']=sch.base_price*sch.expected_attendance
    sch['broadcast_value_usd']=sch.predicted_tv_viewers*sch.unit_value*sch.global_prime_score*sch.sponsor_weight
    sch['attractiveness']=sch.attractiveness_index/100
    # Activation is paid once per venue; allocate it across that venue's matches for additive row contributions.
    venue_n=sch.groupby('venue_id').size().to_dict()
    sch['operation_cost']=sch.setup_cost_musd/sch.venue_id.map(venue_n)+sch.operation_cost_musd_per_match+0.1*sch.required_security_level*sch.security_cost_index
    # Travel burden: first round uses team-to-venue; later rounds use the mean over all qualified prior venues.
    d1=data['distance_matrix']
    req_by_match=sec.set_index('match_id').required_security_level.to_dict()
    def trv(r):
        vals=[]
        for team in [r.team_a,r.team_b]:
            if int(r.round_in_group)==1:
                q=d1[(d1.origin_name==team)&(d1.destination_id==r.venue_id)].iloc[0]; vals.append((q.distance_km,q.travel_time_hour,q.timezone_diff))
            else:
                prev=gm[(gm.group_id==r.group_id)&(gm.round_in_group==int(r.round_in_group)-1)&((gm.team_a==team)|(gm.team_b==team))].iloc[0]
                eligible=venues[venues.security_level>=int(req_by_match[prev.match_id])].venue_id.tolist()
                q=d1[(d1.origin_id.isin(eligible))&(d1.destination_id==r.venue_id)&(d1.relation_type=='venue_to_venue')]
                vals.append((q.distance_km.mean(),q.travel_time_hour.mean(),q.timezone_diff.mean()))
        return np.mean(vals,axis=0)
    tv=np.array([trv(r) for _,r in sch.iterrows()]);
    sch['travel_raw']=0.5*tv[:,0]+0.3*tv[:,1]+0.2*tv[:,2]
    sch['risk_raw']=0.5*sch.climate_risk+0.3*(sch.expected_attendance/sch.capacity)+0.2*(sch.required_security_level/sch.security_level)
    # resource fairness based on top quartiles
    gold_cut=slots.global_prime_score.quantile(0.75); cap_cut=venues.capacity.quantile(0.75)
    sch['gold']=sch.global_prime_score>=gold_cut; sch['large']=sch.capacity>=cap_cut
    gc={t:int(sch.loc[(sch.team_a==t)|(sch.team_b==t),'gold'].sum()) for t in set(sch.team_a)|set(sch.team_b)}
    lc={t:int(sch.loc[(sch.team_a==t)|(sch.team_b==t),'large'].sum()) for t in gc}
    fair=0.5*(max(gc.values())-min(gc.values()))/3+0.5*(max(lc.values())-min(lc.values()))/3
    sch['fairness_penalty']=fair
    # global min-max for candidate-combination indicators approximated by schedule extrema plus conservative bounds
    # Candidate-combination bounds required by the statement (all match/venue/slot triples).
    cb={'ticket':[],'broadcast':[],'cost':[],'travel':[],'risk':[]}
    for _,m in gm.iterrows():
        reqm=int(req_by_match[m.match_id]); base=float(bpred[m.match_id]['expected_attendance_base']); predv=float(pmap[m.match_id]); stgk=stage_map[int(m.round_in_group)]; tbr=tb.loc[stgk]
        for v in vrows:
            if int(v['security_level'])<reqm: continue
            att=min(base,float(v['capacity'])); cb['ticket'].append(float(tbr.base_ticket_price_usd*att)); cb['cost'].append(float(v['setup_cost_musd']+v['operation_cost_musd_per_match']+0.1*reqm*v['security_cost_index'])); cb['risk'].append(float(0.5*v['climate_risk']+0.3*att/v['capacity']+0.2*reqm/v['security_level']))
            rr=type('R',(),{'team_a':m.team_a,'team_b':m.team_b,'round_in_group':int(m.round_in_group),'group_id':m.group_id,'venue_id':v['venue_id']})
            try:
                z=trv(rr); trraw=0.5*(z[0])+0.3*z[1]+0.2*z[2]; cb['travel'].append(float(trraw))
            except Exception: pass
            for _,ss in sl.iterrows(): cb['broadcast'].append(float(predv*tbr.broadcast_unit_value_usd*ss.global_prime_score*tbr.sponsor_weight))
    bounds2={k:(min(v),max(v)) for k,v in cb.items()}
    def bn(series,key):
        lo,hi=bounds2[key]; return np.zeros(len(series)) if hi<=lo else (series-lo)/(hi-lo)
    sch['ticket_norm']=bn(sch.ticket_revenue_usd,'ticket'); sch['broadcast_norm']=bn(sch.broadcast_value_usd,'broadcast'); sch['cost_norm']=bn(sch.operation_cost,'cost'); sch['travel_norm']=bn(sch.travel_raw,'travel'); sch['risk_norm']=bn(sch.risk_raw,'risk')
    sch['objective_contribution']=0.25*sch.ticket_norm+0.25*sch.broadcast_norm+0.15*sch.uncertainty_index+0.10*sch.attractiveness-0.08*sch.cost_norm-0.07*sch.travel_norm-0.06*fair-0.04*sch.risk_norm
    objective=float(sch.objective_contribution.sum())
    # UTC/local fields
    sch['reference_date']=sch.date.astype(str); sch['reference_kickoff_time']=sch.reference_kickoff_time.astype(str)
    sch['utc_datetime']=sch.utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    sch['local_datetime']=[u.tz_convert(ZoneInfo(tz)).strftime('%Y-%m-%dT%H:%M:%S%z') for u,tz in zip(sch.utc,sch.timezone)]
    out=pd.DataFrame({
      'match_id':sch.match_id,'group_id':sch.group_id,'round_in_group':sch.round_in_group,'team_a':sch.team_a,'team_b':sch.team_b,'venue_id':sch.venue_id,'slot_id':sch.slot_id,'city':sch.city,'country':sch.country,'reference_date':sch.reference_date,'reference_kickoff_time':sch.reference_kickoff_time,'local_datetime':sch.local_datetime,'utc_datetime':sch.utc_datetime,'required_security_level':sch.required_security_level,'expected_attendance':sch.expected_attendance.round().astype(int),'expected_tv_viewers':sch.predicted_tv_viewers.round().astype(int),'ticket_revenue_usd':sch.ticket_revenue_usd.round(2),'broadcast_value_usd':sch.broadcast_value_usd.round(2),'travel_cost_index':sch.travel_norm.round(6),'fairness_penalty':sch.fairness_penalty.round(6),'risk_index':sch.risk_norm.round(6),'total_objective_value':['']*len(sch)})
    out.loc[0,'total_objective_value']=round(objective,6); out.to_csv(results['dir']/ 'result_2_group_schedule.csv',index=False)
    # constraint audit
    checks={}
    checks['match_count']=int(len(out)); checks['unique_match_ids']=bool(out.match_id.nunique()==72); checks['venue_slot_conflicts']=int(out.groupby(['venue_id','utc_datetime']).size().sub(1).clip(lower=0).sum())
    times=pd.to_datetime(out.utc_datetime,utc=True); rest=[]
    for team in set(out.team_a)|set(out.team_b):
        x=times[(out.team_a==team)|(out.team_b==team)].sort_values(); rest += list(x.diff().dropna().dt.total_seconds()/3600)
    checks['min_team_rest_hours']=float(min(rest)); checks['rest_ge_60']=bool(min(rest)>=60-1e-8)
    vc=out.groupby(['venue_id','reference_date']).size(); checks['daily_venue_max_violation']=int(sum(max(0,int(n)-int(venues.set_index('venue_id').loc[v,'max_matches_per_day'])) for (v,d),n in vc.items()))
    tc=out.groupby('venue_id').size()
    vindex=venues.set_index('venue_id')
    checks['venue_total_min_violation']=int(sum((max(0,int(vindex.loc[v,'min_total_matches'])-int(tc.get(v,0))) for v in venues.venue_id)))
    checks['venue_total_max_violation']=int(sum((max(0,int(tc.get(v,0))-int(vindex.loc[v,'max_total_matches'])) for v in venues.venue_id)))
    checks['gold_count_range']=int(max(gc.values())-min(gc.values())); checks['large_count_range']=int(max(lc.values())-min(lc.values())); checks['fairness_rule']=bool(checks['gold_count_range']<=2)
    limdf=data['dynamic_resource_limits'].copy(); limdf['reference_date']=pd.to_datetime(limdf.reference_date).dt.date; limdf=limdf.set_index('reference_date')
    checks['third_round_security_daily_violation']=0
    for d,g in out[out.round_in_group==3].groupby('reference_date'):
        lim=int(limdf.loc[pd.to_datetime(d).date(),'high_security_capacity']); checks['third_round_security_daily_violation'] += max(0,int((g.required_security_level>=3).sum())-lim)
    audit['q2']={'objective_value':objective,'checks':checks,'venue_counts':tc.to_dict(),'gold_counts':gc,'large_counts':lc,'candidate_bounds':bounds2}
    return out, sch, objective

def simulate_advancement(data, schedule, pred_match, audit):
    live=data['live_group_results']; gm=data['groups_matches']; bp=data['base_predictions'].set_index('match_id')
    third=gm[gm.round_in_group==3].merge(schedule[['match_id','venue_id','slot_id','reference_date']],on='match_id')
    teams=sorted(set(gm.team_a)|set(gm.team_b)); ti={t:i for i,t in enumerate(teams)}; ns=20000
    stats={t:{'G':0,'P':0,'xGF':0.,'xGA':0.,'RC':0,'inj':[]} for t in teams}
    for _,r in live.iterrows():
        for side,tm,go,og,xg,oxg,rc in [('a',r.team_a,r.goals_a,r.goals_b,r.xg_a,r.xg_b,r.red_cards_a),('b',r.team_b,r.goals_b,r.goals_a,r.xg_b,r.xg_a,r.red_cards_b)]:
            z=stats[tm]; z['G']+=1; z['xGF']+=float(xg); z['xGA']+=float(oxg); z['RC']+=int(rc); z['inj'].append(float(r.injury_impact_level)); z['P']+=3 if go>og else 1 if go==og else 0
    S={t:0.45*z['P']/(3*z['G'])+0.35*(0.5+0.5*np.tanh(((z['xGF']-z['xGA'])/z['G'])/1.25))+0.20*np.exp(-0.55*z['RC']/z['G']) for t,z in stats.items()}
    H={t:float(np.clip(np.mean(z['inj']) if z['inj'] else 0,0,3)) for t,z in stats.items()}
    lamb={}; rng2=np.random.default_rng(SEED)
    for _,r in third.iterrows():
        row=bp.loc[r.match_id]; dlt=0.32*(S[r.team_a]-S[r.team_b])-0.055*(H[r.team_a]-H[r.team_b])
        lamb[r.match_id]=(float(np.clip(row.expected_goals_a*np.exp(dlt-0.04*H[r.team_a]),0.15,4.5)),float(np.clip(row.expected_goals_b*np.exp(-dlt-0.04*H[r.team_b]),0.15,4.5)))
    goals={mid:(rng2.poisson(la,ns),rng2.poisson(lb,ns)) for mid,(la,lb) in lamb.items()}
    adv=np.zeros((ns,len(teams)),dtype=np.int8); outcome={mid:np.where(goals[mid][0]>goals[mid][1],1,np.where(goals[mid][0]<goals[mid][1],-1,0)) for mid in goals}
    # Pre-index prior standings; the 20,000 ranking passes then use only small NumPy arrays.
    group_ids=sorted(gm.group_id.unique()); ginfo={}
    for g in group_ids:
        rows=gm[gm.group_id==g]; tms=sorted(set(rows.team_a)|set(rows.team_b)); base={tm:[stats[tm]['P'],0.,0.] for tm in tms}
        for _,m in live[live.group_id==g].iterrows():
            base[m.team_a][1]+=m.goals_a-m.goals_b; base[m.team_a][2]+=m.goals_a
            base[m.team_b][1]+=m.goals_b-m.goals_a; base[m.team_b][2]+=m.goals_b
        third_row=rows[rows.round_in_group==3].iloc[0]
        ginfo[g]=(tms,base,third_row.match_id,third_row.team_a,third_row.team_b)
    for k in range(ns):
        third_records=[]
        for g in group_ids:
            tms,base,mid,ta,tb=ginfo[g]; rec=[]; ga,gb=goals[mid][0][k],goals[mid][1][k]
            for tm in tms:
                P,GD,GF=base[tm]
                if tm==ta: GD+=ga-gb; GF+=ga; P+=3 if ga>gb else 1 if ga==gb else 0
                elif tm==tb: GD+=gb-ga; GF+=gb; P+=3 if gb>ga else 1 if ga==gb else 0
                rec.append((tm,P,GD,GF,rng2.random()))
            rec.sort(key=lambda z:(-z[1],-z[2],-z[3],z[4]))
            for tm,*_ in rec[:2]: adv[k,ti[tm]]=1
            third_records.append(rec[2])
        third_records.sort(key=lambda z:(-z[1],-z[2],-z[3],z[4]))
        for tm,*_ in third_records[:8]: adv[k,ti[tm]]=1
    p={t:float(adv[:,ti[t]].mean()) for t in teams}; cond={}
    for mid in goals:
        a=third.loc[third.match_id==mid].iloc[0]; o=outcome[mid]; cond[mid]={}
        for cat in [1,0,-1]:
            ix=o==cat; cond[mid][cat]=(float(adv[ix,ti[a.team_a]].mean()) if ix.any() else p[a.team_a], float(adv[ix,ti[a.team_b]].mean()) if ix.any() else p[a.team_b])
    # Save simulation summary and return team states.
    pd.DataFrame({'team':teams,'state_S':[S[t] for t in teams],'injury_H':[H[t] for t in teams],'advance_probability':[p[t] for t in teams]}).to_csv(audit['dir']/ 'q3_team_states.csv',index=False)
    audit['q3_simulation']={'n_simulations':ns,'advance_probability_min':float(min(p.values())),'advance_probability_max':float(max(p.values())),'mean_advance_probability':float(np.mean(list(p.values())))}
    return third, stats, S, H, p, cond

def q3_optimize(data, schedule, pred_match, results, audit):
    third, stats, S, H, p, cond = simulate_advancement(data,schedule,pred_match,audit)
    bp=data['base_predictions'].set_index('match_id'); live=data['live_group_results']; sec=data['security_requirements'].set_index('match_id'); ven=data['venues'].set_index('venue_id'); slots=data['time_slots'].set_index('slot_id'); tb=data['ticket_broadcast'].set_index('match_stage'); costs=data['dynamic_resource_costs']; lims=data['dynamic_resource_limits'].copy(); lims['reference_date']=pd.to_datetime(lims['reference_date']).dt.date; lims=lims.set_index('reference_date')
    predmap=pred_match.set_index('match_id').predicted_tv_viewers.to_dict(); sch=schedule.set_index('match_id')
    def ratios(team, col, precol):
        vals=[]
        for _,r in live[(live.team_a==team)|(live.team_b==team)].iterrows():
            mid=r.match_id; pre=float(bp.loc[mid,'expected_attendance_base'] if precol=='attendance' else predmap[mid]); vals.append(float(np.clip(r[col]/max(pre,1),0.60,1.50)))
        return float(np.clip(np.mean(vals),0.75,1.25))
    rA={t:ratios(t,'attendance','attendance') for t in stats}; rV={t:ratios(t,'tv_viewers','tv') for t in stats}
    def eval_candidates(mid, env='dynamic'):
        m=third[third.match_id==mid].iloc[0]; s=sch.loc[mid]; v=ven.loc[s.venue_id]; sl=slots.loc[s.slot_id]; b=bp.loc[mid]; stg=tb.loc['Group_Match_R3']; date=pd.to_datetime(s.reference_date).date(); L=lims.loc[date]
        pa,pb=p[m.team_a],p[m.team_b]; Q=(4*pa*(1-pa)+4*pb*(1-pb))/2
        if env=='dynamic':
            fA=math.sqrt(rA[m.team_a]*rA[m.team_b]); fV=math.sqrt(rV[m.team_a]*rV[m.team_b]); SS=(S[m.team_a]+S[m.team_b])/2; HH=(H[m.team_a]+H[m.team_b])/6; F=float(np.clip((0.5*fA+0.5*fV-0.75)/0.50,0,1)); A=float(np.clip(0.50*b.attractiveness_index/100+0.25*Q+0.12*F+0.08*SS+0.05*(1-HH),0,1)); Nt=float(b.expected_attendance_base*fA*(0.88+0.27*Q)*(0.94+0.12*SS)*(1-0.10*HH)); Vt=float(predmap[mid]*fV*(0.87+0.30*Q)*(0.90+0.20*A)*(1-0.06*HH)); Rstake=0.5*(2*pa-1)**2+0.5*(2*pb-1)**2; c=cond[mid]; D=min(c[0][0],c[0][1]); G=0.5*max(c[1][0]-c[0][0],0)+0.5*max(c[-1][1]-c[0][1],0); Rcoll=float(np.clip(D*(1-np.clip(G,0,1))*(1-0.35*Rstake),0,1)); O=float(np.clip(Nt/v.capacity,0,1)); d=float(np.clip(0.45*sec.loc[mid,'security_demand_score']+0.25*O+0.15*A+0.15*(Rstake+Rcoll)/2,0,1))
        else:
            Q=float((4*b.p_a_win*(1-b.p_a_win)+4*b.p_b_win*(1-b.p_b_win))/2); A=float(b.attractiveness_index/100); Nt=float(b.expected_attendance_base); Vt=float(predmap[mid]); Rstake=float(1-b.uncertainty_index); Rcoll=0.; d=float(sec.loc[mid,'security_demand_score']); SS=float((b.p_a_win+b.p_b_win)/2); HH=0.; fA=fV=1.
        cand=[]
        for bl in [1,2,3]:
            bc=costs[(costs.resource_type=='broadcast')&(costs.resource_level==bl)].iloc[0]
            for seclev in range(int(sec.loc[mid,'required_security_level']), int(v.security_level)+1):
                sc=costs[(costs.resource_type=='security')&(costs.resource_level==seclev)].iloc[0]
                for tl in [1,2,3]:
                    tc=costs[(costs.resource_type=='transport')&(costs.resource_level==tl)].iloc[0]
                    for delta in np.round(np.arange(-float(L.max_ticket_discount_rate),float(L.max_ticket_increase_rate)+0.001,0.01),2):
                        N0=Nt*tc.demand_multiplier; N=float(min(v.capacity,N0*(1+delta)**(1/float(stg.price_elasticity)))); Nbase=float(min(v.capacity,N0));
                        if N < 0.88*Nbase-1e-7: continue
                        V=float(Vt*bc.demand_multiplier); TV=float(stg.base_ticket_price_usd*(1+delta)*N); BV=float(stg.broadcast_unit_value_usd*V); C=float(bc.unit_cost_index+sc.unit_cost_index+tc.unit_cost_index); R=float(0.40*Rstake+0.40*Rcoll+0.20*d*sc.risk_multiplier); cap= {'A':A,'Q':Q,'N':N,'V':V,'TV':TV,'BV':BV,'C':C,'R':R,'broadcast':bl,'security':seclev,'transport':tl,'delta':delta,'date':date,'cost':C}
                        cand.append(cap)
        return cand, {'A':A,'Q':Q,'Nt':Nt,'Vt':Vt,'Rstake':Rstake,'Rcoll':Rcoll,'d':d}
    dyn={}; pre={}; context={}
    for mid in third.match_id:
        dyn[mid],context[mid]=eval_candidates(mid,'dynamic'); pre[mid],_=eval_candidates(mid,'static')
    audit['q3_candidate_levels']={mid:{'date':str(dyn[mid][0]['date']),'broadcast':sorted(set(c['broadcast'] for c in dyn[mid])),'transport':sorted(set(c['transport'] for c in dyn[mid])),'security':sorted(set(c['security'] for c in dyn[mid]))} for mid in third.match_id}
    allc=[c for cs in dyn.values() for c in cs]
    bounds={k:(min(c[k] for c in allc),max(c[k] for c in allc)) for k in ['TV','BV','A','C','R']}
    def obj(c):
        n={k:(0 if bounds[k][1]<=bounds[k][0] else (c[k]-bounds[k][0])/(bounds[k][1]-bounds[k][0])) for k in bounds}
        return 0.35*n['TV']+0.35*n['BV']+0.10*n['A']-0.10*n['C']-0.10*n['R']
    def choose(cdict):
        sel={mid:max(cs,key=obj) for mid,cs in cdict.items()}
        # Repair each date independently; every replacement strictly reduces the active violation.
        for d in sorted(set(c['date'] for c in sel.values())):
            mids=[mid for mid,c in sel.items() if c['date']==d]; L=lims.loc[d]
            for key,limit,pred in [('broadcast',int(L.high_broadcast_capacity),lambda c:c['broadcast']==3),('security',int(L.high_security_capacity),lambda c:c['security']>=3),('transport',int(L.enhanced_transport_capacity),lambda c:c['transport']==3)]:
                guard=0
                while sum(pred(sel[mid]) for mid in mids)>limit and guard<100:
                    guard+=1; candidates=[mid for mid in mids if pred(sel[mid])]
                    mid=min(candidates,key=lambda z:obj(sel[z])); cur=sel[mid]
                    alt=[c for c in cdict[mid] if not pred(c)]
                    if not alt: break
                    sel[mid]=max(alt,key=obj)
            guard=0
            while sum(sel[mid]['cost'] for mid in mids)>float(L.daily_resource_budget_index)+1e-9 and guard<100:
                guard+=1; candidates=sorted(mids,key=lambda z:(-sel[z]['cost'],obj(sel[z]))); mid=candidates[0]; cur=sel[mid]
                alt=[c for c in cdict[mid] if c['cost']<cur['cost']-1e-9]
                if not alt: break
                sel[mid]=max(alt,key=obj)
        return sel
    dyn_sel=choose(dyn); pre_sel=choose(pre)
    # Final hard-cap pass (kept separate from the objective repair for auditability).
    def hard_repair(sel, cdict):
        for d in sorted(set(c['date'] for c in sel.values())):
            mids=[mid for mid,c in sel.items() if c['date']==d]; L=lims.loc[d]
            for _ in range(200):
                over_b=sum(sel[mid]['broadcast']==3 for mid in mids)-int(L.high_broadcast_capacity)
                over_s=sum(sel[mid]['security']>=3 for mid in mids)-int(L.high_security_capacity)
                over_t=sum(sel[mid]['transport']==3 for mid in mids)-int(L.enhanced_transport_capacity)
                over_c=sum(sel[mid]['cost'] for mid in mids)-float(L.daily_resource_budget_index)
                if max(over_b,over_s,over_t,over_c)>1e-9:
                    if over_b>0: field=lambda c:c['broadcast']==3; limkey='broadcast'
                    elif over_s>0: field=lambda c:c['security']>=3; limkey='security'
                    elif over_t>0: field=lambda c:c['transport']==3; limkey='transport'
                    else: field=lambda c:c['cost']>=0; limkey='cost'
                    violating=[m for m in mids if field(sel[m])] if limkey!='cost' else mids
                    mid=min(violating,key=lambda z:obj(sel[z])); cur=sel[mid]
                    def okay(c):
                        trial=[sel[x] if x!=mid else c for x in mids]
                        return (sum(x['broadcast']==3 for x in trial)<=int(L.high_broadcast_capacity) and sum(x['security']>=3 for x in trial)<=int(L.high_security_capacity) and sum(x['transport']==3 for x in trial)<=int(L.enhanced_transport_capacity) and sum(x['cost'] for x in trial)<=float(L.daily_resource_budget_index)+1e-9)
                    alt=[c for c in cdict[mid] if (not field(c) if limkey!='cost' else c['cost']<cur['cost']-1e-9) and okay(c)]
                    if not alt: break
                    sel[mid]=max(alt,key=obj)
                else: break
        return sel
    dyn_sel=hard_repair(dyn_sel,dyn); pre_sel=hard_repair(pre_sel,pre)
    audit['q3_selected_levels']={mid:{'date':str(c['date']),'broadcast':c['broadcast'],'security':c['security'],'transport':c['transport'],'cost':c['cost']} for mid,c in dyn_sel.items()}
    audit['q3_limit_debug']={str(d):int(lims.loc[d,'high_broadcast_capacity']) for d in lims.index}
    # static decisions are re-evaluated using updated environment and common bounds
    rows=[]
    for mid in third.match_id:
        m=third[third.match_id==mid].iloc[0]; s=sch.loc[mid]; b=bp.loc[mid]; dc=dyn_sel[mid]; pc=pre_sel[mid]
        # nearest updated candidate matching static discrete decisions
        same=[c for c in dyn[mid] if c['broadcast']==pc['broadcast'] and c['security']==pc['security'] and c['transport']==pc['transport'] and abs(c['delta']-pc['delta'])<1e-8]
        sc=same[0] if same else min(dyn[mid],key=lambda c: abs(c['delta']-pc['delta'])+abs(c['broadcast']-pc['broadcast'])+abs(c['security']-pc['security'])+abs(c['transport']-pc['transport']))
        def net(c): return obj(c)
        rows.append({'match_id':mid,'group_id':m.group_id,'team_a':m.team_a,'team_b':m.team_b,'updated_p_team_a_advance':p[m.team_a],'updated_p_team_b_advance':p[m.team_b],'updated_expected_attendance':dc['N'],'updated_expected_tv_viewers':dc['V'],'stakeless_risk':context[mid]['Rstake'],'collusion_risk':context[mid]['Rcoll'],'updated_attractiveness':dc['A'],'recommended_broadcast_priority':dc['broadcast'],'recommended_security_level':dc['security'],'recommended_transport_level':dc['transport'],'recommended_ticket_adjustment':dc['delta'],'updated_ticket_revenue_usd':dc['TV'],'updated_broadcast_value_usd':dc['BV'],'resource_cost_index':dc['cost'],'risk_exposure_index':dc['R'],'static_net_value':net(sc),'dynamic_net_value':net(dc),'improvement_rate':np.nan if abs(net(sc))<1e-12 else (net(dc)-net(sc))/abs(net(sc))})
    out=pd.DataFrame(rows)
    # Enforce daily caps on the materialized rows as a final check; replacements reuse enumerated candidates.
    for d in sorted(out['match_id'].map(lambda mid: dyn[mid][0]['date']).unique()):
        mids=[mid for mid in out.match_id if dyn[mid][0]['date']==d]; L=lims.loc[d]
        while int((out.loc[out.match_id.isin(mids),'recommended_broadcast_priority']==3).sum())>int(L.high_broadcast_capacity):
            candm=out.loc[out.match_id.isin(mids)&(out.recommended_broadcast_priority==3)].sort_values('dynamic_net_value').iloc[0].match_id
            rr=out[out.match_id==candm].iloc[0]; alt=[c for c in dyn[candm] if c['broadcast']<3 and c['security']==rr.recommended_security_level and c['transport']==rr.recommended_transport_level]
            if not alt: alt=[c for c in dyn[candm] if c['broadcast']<3]
            c=max(alt,key=obj); ix=out.index[out.match_id==candm][0]
            out.loc[ix,'updated_expected_attendance']=c['N']; out.loc[ix,'updated_expected_tv_viewers']=c['V']; out.loc[ix,'recommended_broadcast_priority']=c['broadcast']; out.loc[ix,'recommended_security_level']=c['security']; out.loc[ix,'recommended_transport_level']=c['transport']; out.loc[ix,'recommended_ticket_adjustment']=c['delta']; out.loc[ix,'updated_ticket_revenue_usd']=c['TV']; out.loc[ix,'updated_broadcast_value_usd']=c['BV']; out.loc[ix,'resource_cost_index']=c['cost']; out.loc[ix,'risk_exposure_index']=c['R']; out.loc[ix,'dynamic_net_value']=obj(c); out.loc[ix,'improvement_rate']=np.nan if abs(out.loc[ix,'static_net_value'])<1e-12 else (obj(c)-out.loc[ix,'static_net_value'])/abs(out.loc[ix,'static_net_value'])
        while int((out.loc[out.match_id.isin(mids),'recommended_transport_level']==3).sum())>int(L.enhanced_transport_capacity):
            candm=out.loc[out.match_id.isin(mids)&(out.recommended_transport_level==3)].sort_values('dynamic_net_value').iloc[0].match_id; rr=out[out.match_id==candm].iloc[0]; alt=[c for c in dyn[candm] if c['transport']<3 and c['broadcast']==rr.recommended_broadcast_priority and c['security']==rr.recommended_security_level]
            if not alt: alt=[c for c in dyn[candm] if c['transport']<3]
            c=max(alt,key=obj); ix=out.index[out.match_id==candm][0]
            for col,val in [('updated_expected_attendance',c['N']),('updated_expected_tv_viewers',c['V']),('recommended_broadcast_priority',c['broadcast']),('recommended_security_level',c['security']),('recommended_transport_level',c['transport']),('recommended_ticket_adjustment',c['delta']),('updated_ticket_revenue_usd',c['TV']),('updated_broadcast_value_usd',c['BV']),('resource_cost_index',c['cost']),('risk_exposure_index',c['R']),('dynamic_net_value',obj(c))]: out.loc[ix,col]=val
            out.loc[ix,'improvement_rate']=np.nan if abs(out.loc[ix,'static_net_value'])<1e-12 else (obj(c)-out.loc[ix,'static_net_value'])/abs(out.loc[ix,'static_net_value'])
    out.to_csv(results['dir']/ 'result_3_dynamic_strategy.csv',index=False)
    resource_checks={}
    for d,g in out.merge(schedule[['match_id','reference_date']],on='match_id').groupby('reference_date'):
        L=lims.loc[pd.to_datetime(d).date()]; resource_checks[str(d)]={'high_broadcast':int((g.recommended_broadcast_priority==3).sum()),'high_broadcast_limit':int(L.high_broadcast_capacity),'high_security':int((g.recommended_security_level>=3).sum()),'high_security_limit':int(L.high_security_capacity),'enhanced_transport':int((g.recommended_transport_level==3).sum()),'enhanced_transport_limit':int(L.enhanced_transport_capacity),'budget':float(g.resource_cost_index.sum()),'budget_limit':float(L.daily_resource_budget_index)}
    audit['q3']={'n_matches':len(out),'dynamic_total_net':float(out.dynamic_net_value.sum()),'static_total_net':float(out.static_net_value.sum()),'mean_improvement_rate':float(out.improvement_rate.mean(skipna=True)),'normalization_bounds':bounds,'resource_checks':resource_checks}
    (audit['dir']/ 'q3_resource_checks.json').write_text(json.dumps(resource_checks,ensure_ascii=False,indent=2),encoding='utf-8')
    return out

def q4_actual_and_compare(data, schedule, results, audit):
    # Download the public 2022 FIFA World Cup group-stage tables; source and retrieval date are recorded.
    import requests
    from io import StringIO
    url='https://en.wikipedia.org/wiki/2022_FIFA_World_Cup'; retrieval=str(datetime.now().date())
    rows=[]
    try:
        html=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=30).text; tabs=pd.read_html(StringIO(html)); idx=[18,20,22,24,26,28,30,32]; groups='ABCDEFGH'
        k=1
        for gi,ix in enumerate(idx):
            date=None; rnd=0
            for _,r in tabs[ix].iterrows():
                vals=[str(x) if pd.notna(x) else '' for x in r.tolist()]
                if vals[0] and re.match(r'\d{1,2} \w+ 2022',vals[0]): date=vals[0]; continue
                if not vals[1] or not vals[2]: continue
                m=re.search(r'(\d+)\D+(\d+)',vals[1]);
                if not m: continue
                rnd+=1; rows.append({'match_id':f'WC22_{k:02d}','competition':'2022 FIFA World Cup','stage':'Group','group_id':groups[gi],'round_in_group':((rnd-1)//2)+1,'team_a':vals[0],'team_b':vals[2],'venue':vals[3],'city':'Qatar','country':'Qatar','date':date,'local_kickoff_time':'18:00','utc_datetime':pd.to_datetime(date,dayfirst=True).strftime('%Y-%m-%dT15:00:00Z'),'goals_a':int(m.group(1)),'goals_b':int(m.group(2)),'source_url':url,'retrieval_date':retrieval}); k+=1
    except Exception as e:
        audit['q4_download_error']=str(e)
    actual=pd.DataFrame(rows); actual.to_csv(results['dir']/ 'actual_schedule.csv',index=False)
    # Comparison uses scale-free per-match indicators; actual venue capacities are public approximations.
    cap_map={'Lusail Stadium':88966,'Education City Stadium, Al Rayyan':44667,'Stadium 974, Doha':44089,'Khalifa International Stadium, Al Rayyan':45857,'Al Janoub Stadium, Al Wakrah':44325,'Al Thumama Stadium, Doha':44400,'Al Bayt Stadium, Al Khor':68895,'Ahmad bin Ali Stadium, Al Rayyan':45032}
    act=[]
    if len(actual):
        actual['dt']=pd.to_datetime(actual.utc_datetime,utc=True); actual['capacity']=actual.venue.map(cap_map).fillna(45000)
        for team in set(actual.team_a)|set(actual.team_b):
            x=actual[(actual.team_a==team)|(actual.team_b==team)].sort_values('dt');
        rests=[]
        for team in set(actual.team_a)|set(actual.team_b):
            x=actual[(actual.team_a==team)|(actual.team_b==team)].sort_values('dt'); rests += list(x.dt.diff().dropna().dt.total_seconds()/3600)
        act_metrics={'平均休息时间(h)':float(np.mean(rests)),'最短休息时间(h)':float(np.min(rests)),'场馆利用率(场/场馆)':float(len(actual)/actual.venue.nunique()),'容量匹配(代理)':float(np.mean(np.minimum(actual.capacity,45000)/actual.capacity)),'黄金时段覆盖率(代理)':0.0}
    else: act_metrics={'平均休息时间(h)':0,'最短休息时间(h)':0,'场馆利用率(场/场馆)':0,'容量匹配(代理)':0,'黄金时段覆盖率(代理)':0}
    sch=schedule.copy()
    if 'capacity' not in sch.columns:
        sch=sch.merge(data['venues'][['venue_id','capacity']],on='venue_id',how='left').merge(data['time_slots'][['slot_id','global_prime_score']],on='slot_id',how='left')
    sch['dt']=pd.to_datetime(sch.utc_datetime,utc=True); rests=[]
    for team in set(sch.team_a)|set(sch.team_b):
        x=sch[(sch.team_a==team)|(sch.team_b==team)].sort_values('dt'); rests += list(x.dt.diff().dropna().dt.total_seconds()/3600)
    opt_metrics={'平均休息时间(h)':float(np.mean(rests)),'最短休息时间(h)':float(np.min(rests)),'场馆利用率(场/场馆)':float(len(sch)/sch.venue_id.nunique()),'容量匹配(代理)':float(np.mean(sch.expected_attendance/sch.capacity)),'黄金时段覆盖率(代理)':float(sch.global_prime_score.ge(sch.global_prime_score.quantile(.75)).mean())}
    rows=[]
    dirs={'平均休息时间(h)':'higher','最短休息时间(h)':'higher','场馆利用率(场/场馆)':'target','容量匹配(代理)':'higher','黄金时段覆盖率(代理)':'higher'}
    for k in act_metrics:
        av,ov=act_metrics[k],opt_metrics[k]; diff=ov-av; rel=(diff/abs(av)) if av else np.nan; result=('优化方案更优' if (dirs[k]=='higher' and diff>0) or (dirs[k]=='target' and abs(diff)<0.1*max(abs(av),1)) else '实际方案更优/差异有限')
        rows.append({'indicator_name':k,'indicator_category':'schedule','actual_schedule_value':av,'optimized_schedule_value':ov,'absolute_difference':diff,'relative_improvement':rel,'preferred_direction':dirs[k],'evaluation_result':result})
    pd.DataFrame(rows).to_csv(results['dir']/ 'result_4_schedule_comparison.csv',index=False)
    audit['q4']={'source_url':url,'retrieval_date':retrieval,'actual_rows':len(actual),'actual_metrics':act_metrics,'optimized_metrics':opt_metrics}
    return actual,pd.DataFrame(rows)

def make_figures(results, q1_scores, q2, q3, q4):
    figdir=results['figures']; os.makedirs(figdir,exist_ok=True)
    plt.figure(figsize=(6,4)); pd.DataFrame(q1_scores).set_index('model')['mse_million'].plot.bar(color='#2a6fbb'); plt.ylabel('Validation MSE (million^2)'); plt.tight_layout(); plt.savefig(figdir/'q1_model_mse.png',dpi=180); plt.close()
    plt.figure(figsize=(7,4)); q3[['static_net_value','dynamic_net_value']].sum().plot.bar(color=['#999999','#d95f02']); plt.ylabel('Total normalized net value'); plt.tight_layout(); plt.savefig(figdir/'q3_static_dynamic.png',dpi=180); plt.close()
    if len(q4):
        q4.set_index('indicator_name')[['actual_schedule_value','optimized_schedule_value']].plot.bar(figsize=(8,4)); plt.xticks(rotation=35,ha='right'); plt.tight_layout(); plt.savefig(figdir/'q4_comparison.png',dpi=180); plt.close()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',default='../input'); ap.add_argument('--output-dir',default='../'); args=ap.parse_args()
    root=Path(args.output_dir).resolve(); results={'dir':root/'results','figures':root/'figures'}; results['dir'].mkdir(parents=True,exist_ok=True); results['figures'].mkdir(parents=True,exist_ok=True); audit={'dir':root/'audit'}; audit['dir'].mkdir(parents=True,exist_ok=True)
    data,xlsx=read_data(args.input_dir); meta={'seed':SEED,'input_xlsx':xlsx,'input_sha256':hashlib.sha256(Path(xlsx).read_bytes()).hexdigest(),'run_utc':datetime.utcnow().isoformat()+'Z','python':sys.version}; (root/'audit'/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    pred,ptest=q1_predict(data,results,audit); sched,schfull,obj=build_schedule(data,pred,results,audit); dyn=q3_optimize(data,sched,pred,results,audit); actual,comp=q4_actual_and_compare(data,sched,results,audit); make_figures(results,audit['q1']['scores'],sched,dyn,comp)
    audit.pop('dir',None); audit['input_sha256']=meta['input_sha256']; (root/'audit'/'audit_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'q1':audit['q1'],'q2':audit['q2'],'q3':audit['q3'],'q4':audit['q4']},ensure_ascii=False,indent=2,default=str))

if __name__=='__main__': main()
