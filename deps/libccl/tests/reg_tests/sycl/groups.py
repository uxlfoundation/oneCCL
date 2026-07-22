import re
import sys

def check_group_overlap(file_path):
    # Regex to capture group and timestamp information
    log_pattern = re.compile(r"Group\[(\d+)\] => (Starting chunk \d+|Finished) => (\d+)")

    # Dictionary to store ongoing groups with their start and end times
    group_timings = {}

    # List of errors to record any overlap
    errors = []

    with open(file_path, 'r') as file:
        for line in file:
            match = log_pattern.search(line)
            if match:
                group_id, action, timestamp = match.groups()
                timestamp = int(timestamp)
                
                if action.startswith("Starting chunk "):
                    # Record the earliest start time of the group
                    if group_id not in group_timings or \
                       (group_timings[group_id]['start'] > timestamp and group_timings[group_id]['end'] is None):
                        group_timings[group_id] = {'start': timestamp, 'end': None}
                elif action == 'Finished':
                    # Update the end time of the group
                    group_timings[group_id]['end'] = timestamp

    if len(group_timings.items()) == 0:
        print(f"No timings were found in log({file_path})")
        exit(1)

    for group_id, timings in group_timings.items():
        start_a, end_a = timings['start'], timings['end']
        print(f"Group[{group_id}]:\t{start_a}\t{end_a}\t")

    # Check for overlaps
    for group_id, timings in group_timings.items():
        for other_group_id, other_timings in group_timings.items():
            if group_id != other_group_id:
                # Check for intersection
                if (timings['start'] < other_timings.get('end', float('inf')) and
                    other_timings['start'] < timings.get('end', float('inf'))):
                    errors.append(f"Overlap detected between Group[{group_id}] and Group[{other_group_id}]")

    if errors:
        print("Overlaps found:")
        for error in errors:
            print(error)
        exit(1)
    else:
        print("No overlaps found.")

if len(sys.argv) != 2:
    print("The script requires a file with timestamps to analyze!")
    exit(1)

    
check_group_overlap(sys.argv[1])

