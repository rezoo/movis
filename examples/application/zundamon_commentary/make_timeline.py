from pathlib import Path

import pandas as pd
from movis.contrib.voicevox import make_timeline_from_voicevox, merge_timeline


def main():
    timeline_path = Path('outputs/timeline.tsv')
    Path('outputs').mkdir(exist_ok=True)
    timeline = make_timeline_from_voicevox(
        'audio', extra_columns=(("slide", 0), ("status", "n"), ("section", "")))
    if timeline_path.exists():
        timeline = merge_timeline(
            pd.read_csv(timeline_path, sep='\t', na_filter=False),
            timeline, key='hash')
    timeline.to_csv('outputs/timeline.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
