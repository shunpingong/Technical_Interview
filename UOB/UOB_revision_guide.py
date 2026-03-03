"""
UOB IA — Python Data Revision Guide
Run this script in a Jupyter notebook cell or as a .py file for interactive practice.
Sections:
 1) Python basics refresher
 2) Pandas fundamentals
 3) Grouping, aggregation, KPIs
 4) Joins and reshaping
 5) Simple intelligent automation examples
 6) Optional light ML sketch (requires scikit-learn)

All datasets are small, in-memory, and use realistic column names: call_id, timestamp, queue_time_sec, handle_time_sec, agent_id, resolved, abandoned, atm_id, withdrawal_amount, etc.
"""

# SECTION 1: Python basics refresher
# Variables and basic types
a = 10            # int
b = 3.14          # float
c = 'hello'       # str
d = True          # bool
print(type(a), type(b), type(c), type(d))  # <class 'int'> ...

# Lists, dicts, tuples, sets
my_list = [1, 2, 3]
my_dict = {'agent_id': 'A01', 'calls': 12}
my_tuple = (10, 20)
my_set = {1, 2, 3}
print('list/dict/tuple/set:', my_list, my_dict, my_tuple, my_set)

# for loops and list comprehensions
squared = []
for x in range(5):
    squared.append(x * x)
squared2 = [x * x for x in range(5)]
print('squared', squared2)

# conditionals
score = 72
if score >= 80:
    grade = 'A'
elif score >= 60:
    grade = 'B'
else:
    grade = 'C'
print('grade=', grade)

# functions and return values
def avg(numbers):
    """Return average or None for empty list."""
    return sum(numbers) / len(numbers) if numbers else None

print('avg example:', avg([10, 20, 30]))

# Small practice tasks
print('\n-- Small practice tasks --')
# 1) Compute average call duration
durations = [45, 30, 120, 60]
print('average duration (s):', avg(durations))

# 2) Filter names starting with 'A'
names = ['Alice', 'Bob', 'Andrew', 'Zoe']
a_names = [n for n in names if n.startswith('A')]
print('names starting with A:', a_names)

# 3) Word count
txt = 'customer called and call was transferred'
print('word count:', len(txt.split()))


# SECTION 2: Pandas fundamentals
import pandas as pd
import numpy as np

print('\n-- Pandas basics: create small contact centre table --')
# Create contact centre DataFrame
calls = pd.DataFrame([
    {'call_id': 'C001', 'timestamp': '2025-12-01 08:05:00', 'queue_time_sec': 10, 'handle_time_sec': 300, 'agent_id': 'A01', 'resolved': True, 'abandoned': False},
    {'call_id': 'C002', 'timestamp': '2025-12-01 08:12:00', 'queue_time_sec': 200, 'handle_time_sec': 0, 'agent_id': None, 'resolved': False, 'abandoned': True},
    {'call_id': 'C003', 'timestamp': '2025-12-01 09:01:00', 'queue_time_sec': 20, 'handle_time_sec': 180, 'agent_id': 'A02', 'resolved': True, 'abandoned': False},
    {'call_id': 'C004', 'timestamp': '2025-12-01 09:05:00', 'queue_time_sec': 120, 'handle_time_sec': 240, 'agent_id': 'A01', 'resolved': False, 'abandoned': False},
])

# dtypes and datetime
calls['timestamp'] = pd.to_datetime(calls['timestamp'])
calls['queue_time_sec'] = calls['queue_time_sec'].astype(int)
calls['handle_time_sec'] = calls['handle_time_sec'].astype(int)
# add date parts
calls['hour'] = calls['timestamp'].dt.hour
calls['weekday'] = calls['timestamp'].dt.day_name()

print('\ncalls.head():')
print(calls.head().to_string(index=False))

print('\ninfo:')
print(calls.info())

print('\ndescribe:')
print(calls.describe(include='all'))

print('\nagent counts:')
print(calls['agent_id'].value_counts(dropna=False))

# Selecting columns/rows
print('\nselect call_id list:')
print(calls['call_id'].tolist())

print('\nselect call_id and queue_time_sec:')
print(calls[['call_id', 'queue_time_sec']])

print('\nfilter long queues > 60s:')
long_queue = calls[calls['queue_time_sec'] > 60]
print(long_queue[['call_id', 'queue_time_sec']])

# Missing data
print('\nmissing counts:')
print(calls.isna().sum())

print('\nfill missing agent_id with UNASSIGNED:')
calls_filled = calls.fillna({'agent_id': 'UNASSIGNED'})
print(calls_filled[['call_id', 'agent_id']])

# drop example (non-destructive)
print('\ndrop rows where timestamp missing (none in this sample):')
print(calls.dropna(subset=['timestamp']))

# Changing dtypes already shown above


# SECTION 3: Grouping, aggregation and KPIs
print('\n-- KPIs and groupby examples --')
# Abandonment by hour
g = calls.groupby('hour').agg(
    total_calls=('call_id', 'count'),
    abandoned_calls=('abandoned', lambda x: x.sum()),
    avg_queue=('queue_time_sec', 'mean')
)
g['abandon_rate'] = g['abandoned_calls'] / g['total_calls']
print('\nabandonment by hour:')
print(g.reset_index())

# Average handle time by agent (exclude abandoned calls because handle_time is 0)
by_agent = calls[~calls['abandoned']].groupby('agent_id').agg(
    handled_calls=('call_id', 'count'),
    avg_handle=('handle_time_sec', 'mean')
)
print('\navg handle time by agent:')
print(by_agent.reset_index())

# SLA breach: queue_time_sec > 120
calls['sla_breach'] = calls['queue_time_sec'] > 120
sla = calls.groupby('hour').agg(total=('call_id', 'count'), breaches=('sla_breach', 'sum'))
sla['breach_rate'] = sla['breaches'] / sla['total']
print('\nSLA breaches by hour:')
print(sla.reset_index())

# Exercise: compute abandonment rate by agent (tiny exercise)
print('\nExercise: abandonment rate by agent (hint: groupby agent_id, sum abandoned, count)')
ab_by_agent = calls.groupby('agent_id').agg(total=('call_id', 'count'), abandoned=('abandoned', 'sum'))
ab_by_agent['abandon_rate'] = ab_by_agent['abandoned'] / ab_by_agent['total']
print(ab_by_agent.reset_index())


# SECTION 4: Joins and reshaping
print('\n-- Joins and pivot tables --')
atm_master = pd.DataFrame([
    {'atm_id': 'ATM01', 'location': 'Orchard', 'capacity': 200000, 'last_topup': '2025-11-30'},
    {'atm_id': 'ATM02', 'location': 'Tampines', 'capacity': 150000, 'last_topup': '2025-11-29'},
])
transactions = pd.DataFrame([
    {'txn_id': 'T001', 'atm_id': 'ATM01', 'date': '2025-12-01', 'withdrawal_amount': 200},
    {'txn_id': 'T002', 'atm_id': 'ATM01', 'date': '2025-12-01', 'withdrawal_amount': 500},
    {'txn_id': 'T003', 'atm_id': 'ATM02', 'date': '2025-12-01', 'withdrawal_amount': 1000},
])

# Merge to add ATM info to transactions
tx = transactions.merge(atm_master, on='atm_id', how='left')
print('\nmerged transactions:')
print(tx)

# Pivot: sum of withdrawals by location
pivot = tx.pivot_table(index='location', values='withdrawal_amount', aggfunc='sum')
print('\nwithdrawal sum by location:')
print(pivot)

# Exercise: left vs inner join demonstration
print('\nExercise: inner join between transactions and atm_master (should be same rows here):')
print(transactions.merge(atm_master, on='atm_id', how='inner'))


# SECTION 5: Simple intelligent automation examples
print('\n-- Simple automation rules --')
# Flag calls needing attention: long queue or zero handle_time
calls['needs_attention'] = (calls['queue_time_sec'] > 120) | (calls['handle_time_sec'] == 0)
print(calls[['call_id', 'queue_time_sec', 'handle_time_sec', 'needs_attention']])

# ATM top-up rule: sum withdrawals > capacity * 0.8
atm_tx_sum = tx.groupby('atm_id').withdrawal_amount.sum().reset_index().rename(columns={'withdrawal_amount': 'sum_withdrawn'})
atm_status = atm_master.merge(atm_tx_sum, on='atm_id', how='left').fillna({'sum_withdrawn': 0})
atm_status['topup_needed'] = atm_status['sum_withdrawn'] > (atm_status['capacity'] * 0.8)
print('\nATM status and top-up needed:')
print(atm_status)

# Wrap logic in functions for reuse
def flag_calls_for_review(df, queue_threshold=120, zero_handle=True):
    df2 = df.copy()
    df2['flag'] = (df2['queue_time_sec'] > queue_threshold) | ((df2['handle_time_sec'] == 0) & zero_handle)
    return df2[df2['flag']]

print('\nflagged calls (threshold 60s):')
print(flag_calls_for_review(calls, queue_threshold=60)[['call_id', 'queue_time_sec', 'flag']])

# Small scheduling rule: create alerts for ATMs needing topup
def atm_topup_alerts(atm_master_df, tx_df, threshold_fraction=0.8):
    txsum = tx_df.groupby('atm_id').withdrawal_amount.sum().reset_index().rename(columns={'withdrawal_amount': 'sum_withdrawn'})
    joined = atm_master_df.merge(txsum, on='atm_id', how='left').fillna({'sum_withdrawn': 0})
    joined['topup_needed'] = joined['sum_withdrawn'] > (joined['capacity'] * threshold_fraction)
    return joined[joined['topup_needed']]

print('\nATMs flagged for top-up:')
print(atm_topup_alerts(atm_master, transactions))


# SECTION 6: Optional light ML sketch (only if scikit-learn is available)
print('\n-- Optional ML sketch --')
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    df_ml = calls.copy()
    df_ml['abandoned_int'] = df_ml['abandoned'].astype(int)
    X = df_ml[['queue_time_sec', 'hour']]
    y = df_ml['abandoned_int']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.4)
    model = LogisticRegression().fit(X_train, y_train)
    print('ML score (accuracy):', model.score(X_test, y_test))
except Exception as e:
    print('scikit-learn not available or error:', e)
    print('To run this section: pip install scikit-learn')


# Appendix: Case management example for operations KPIs
print('\n-- Case management example --')
cases = pd.DataFrame([
    {'case_id': 'CA001', 'created_time': '2025-11-20 09:00', 'closed_time': '2025-11-20 12:00', 'status': 'Closed', 'team': 'Onboarding', 'priority': 'High'},
    {'case_id': 'CA002', 'created_time': '2025-11-21 10:00', 'closed_time': None, 'status': 'Open', 'team': 'Onboarding', 'priority': 'Low'},
    {'case_id': 'CA003', 'created_time': '2025-11-19 08:00', 'closed_time': '2025-11-19 09:30', 'status': 'Closed', 'team': 'Support', 'priority': 'Medium'},
])

cases['created_time'] = pd.to_datetime(cases['created_time'])
cases['closed_time'] = pd.to_datetime(cases['closed_time'])
cases['resolution_hours'] = (cases['closed_time'] - cases['created_time']).dt.total_seconds() / 3600
print(cases[['case_id', 'team', 'status', 'resolution_hours']])
print('\navg resolution hours by team:')
print(cases.groupby('team').resolution_hours.mean())

print('\n--- End of revision guide ---')
