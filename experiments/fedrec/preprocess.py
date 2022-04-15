"""
    Preprocess MIND data into FLUTE-compatible format.
    --input: folder containing a MIND dataset
    --time_split: what fraction of data to set aside for pretraining, based on timestamp  (default 0.1).
    
    Outputs:
    'behaviors_pretraining.tsv': list of behaviors in the pretraining set (prior to the timestamp determined by time_split). This is in the same format as behaviors.tsv.
    'behaviors_flute.json': flute-formatted json file containing the training data:
    { 'users': [list of user ids in training, each user appearing once],
        'num_samples': [list, each entry corresponding to the users in 'users', indicating how many samples that user has ],
        'user_data': dictionary keyed by user id, containing the user's data in this format:
        { '<user-id>': {
                'in_pretraining': <boolean flag indicating if the user is also part of the pretraining set>,
                'x': [list of dictionary records for the user, each corresponding to a single row from behaviors.tsv.  only records falling after the split timestamp are here.]
            }
        }
        eg
        "U80234": {
            "in_pretraining": false,
            "x": [
                {
                    "ImpressionID": "1",
                    "UserID": "U80234",
                    "Time": "11/15/2019 12:37:50 PM",
                    "History": "N55189 N46039 N51741 N53234 N11276 N264 N40716 N28088 N43955 N6616 N47686 N63573 N38895 N30924 N35671",
                    "Impressions": "N28682-0 N48740-0 N31958-1 N34130-0 N6916-0 N5472-0 N50775-0 N24802-0 N19990-0 N33176-0 N62365-0 N5940-0 N6400-0 N58098-0 N42844-0 N49285-0 N51470-0 N53572-0 N11930-0 N21679-0 N55237-0 N29862-0"
                },
                {
                    "ImpressionID": "16838",
                    "UserID": "U80234",
                    "Time": "11/15/2019 8:34:00 AM",
                    "History": "N55189 N46039 N51741 N53234 N11276 N264 N40716 N28088 N43955 N6616 N47686 N63573 N38895 N30924 N35671",
                    "Impressions": "N58656-0 N29490-0 N58748-0 N16120-0 N38901-1 N28072-0 N14478-0 N13854-0 N19990-0 N27289-0 N16344-0 N57560-0 N12320-0 N7556-0 N32734-0 N13408-0 N36940-0 N63342-0 N6432-0 N6916-0 N52514-0 N6645-0 N9284-1 N13270-0 N46976-0 N56458-0 N37055-0 N2852-0 N57327-0 N29393-0 N32288-1 N61811-0 N20036-0 N52103-0 N16237-0 N42478-0 N58612-0 N23336-0 N496-0 N54277-0 N40299-0 N28345-0 N57007-0 N39907-0 N12446-0 N42844-0 N43277-0 N5051-0 N52850-0 N7342-0 N50894-0 N61829-0 N23692-0 N11930-0 N37233-0 N58251-0 N926-0 N36779-0 N14637-0 N23490-0 N57359-0 N32708-0 N18971-0 N22257-0 N45289-0 N14056-0 N5940-0 N58098-0 N32237-0 N19611-0 N53283-0 N26572-0 N6638-0 N14507-0 N30290-0 N17031-0 N51793-0 N44558-0 N55712-0 N23629-0 N52323-0 N6194-0 N59602-0 N46162-0 N62805-0 N61697-0 N60939-0 N31958-0 N41946-0 N33397-0 N37338-0 N7993-0 N23355-0 N23513-0 N35815-0"
                },
            ]
        }
"""
import argparse
import csv
import os
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, required=False, default='MINDsmall_dev')
parser.add_argument('--time_split', type=float, default=0.1, required=False, help='Fraction of time for pretraining')

args = parser.parse_args()

output = args.input + '.json'

out_json = {'users': [], 'num_samples': None, 'user_data': {}}
timestamps = []
rows = []
user_timestamps = []
user_pretraining = set()
all_users = set()
header = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']

with open(os.path.join(args.input, 'behaviors.tsv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        rows.append(row)
        d = dict(zip(header, row))
        timestamp = datetime.strptime(d['Time'], '%m/%d/%Y %I:%M:%S %p')
        timestamps.append(timestamp)
        user_timestamps.append((d['UserID'], timestamp))
        all_users.add(d['UserID'])

split_time = list(sorted(timestamps))[int(len(timestamps) * args.time_split)]

for u, t in user_timestamps:
    if t < split_time:
        user_pretraining.add(u)

# On Windows I had to add newline=''.  Double check this works on linux.
with open(os.path.join(args.input, 'behaviors_pretraining.tsv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    num_samples = {}
    for row in rows:
        d = dict(zip(header, row))
        timestamp = datetime.strptime(d['Time'], '%m/%d/%Y %I:%M:%S %p')
        if timestamp < split_time:
            writer.writerow(row)
        else:
            u = d['UserID']
            if u not in out_json['user_data']:
                out_json['users'].append(u)
                num_samples[u] = 0
                out_json['user_data'][u] = {
                    'in_pretraining': u in user_pretraining,
                    'x': []
                }
            out_json['user_data'][u]['x'].append(d)
            num_samples[u] += 1
    out_json['num_samples'] = [num_samples[u] for u in out_json['users']]
with open(os.path.join(args.input, 'behaviors_flute.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(out_json, indent='\t') + '\n')

logging.info('All users: {}'.format(len(all_users)))
logging.info('Pretraining users: {}'.format(len(user_pretraining)))
logging.info('Training Users: {}'.format(len(out_json['users'])))
