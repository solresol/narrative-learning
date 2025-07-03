import json
from modules.postgres import get_connection

SPAN_LIMIT = '28 days'


def main():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT dataset, config_file FROM datasets")
    datasets = cur.fetchall()

    issues = []
    earliest_dates = []

    for dataset, cfg_path in datasets:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        table = cfg.get('rounds_table', f"{dataset}_rounds")

        cur.execute(f"SELECT MIN(round_start) FROM {table}")
        earliest = cur.fetchone()[0]
        earliest_dates.append(earliest)

        cur.execute(
            f"""
            SELECT investigation_id, MIN(round_start) AS first_round,
                   MAX(round_start) AS last_round,
                   MAX(round_start) - MIN(round_start) AS span,
                   COUNT(*) AS round_count
              FROM {table}
             GROUP BY investigation_id
            HAVING COUNT(*) > 1 AND MAX(round_start) - MIN(round_start) > INTERVAL '{SPAN_LIMIT}'
             ORDER BY span DESC
            """
        )
        rows = cur.fetchall()
        if rows:
            issues.append((dataset, rows))

    print('Earliest round_start in any dataset:', min(earliest_dates))
    if not issues:
        print('No investigations have rounds more than', SPAN_LIMIT, 'apart.')
    else:
        for dataset, rows in issues:
            print(f"Dataset {dataset} investigations with rounds spaced over {SPAN_LIMIT}:")
            for inv_id, first, last, span, count in rows:
                print(f"  investigation {inv_id}: {count} rounds from {first} to {last} span {span}")

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
