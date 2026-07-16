import json
from pathlib import Path

def test_index_exists_and_has_expected_fields():
    idx_path = Path('dti_spectra_index.json')
    assert idx_path.exists(), "dti_spectra_index.json must exist at repo root"
    idx = json.loads(idx_path.read_text())
    assert isinstance(idx, dict), "index must be a JSON object mapping sha->entry"
    # check at least one entry has required keys
    required = {'filename','sha256','n','edges','k_chosen','timestamp','outputs'}
    found = False
    for sha, entry in idx.items():
        if required.issubset(set(entry.keys())):
            found = True
            # minimal sanity checks
            assert isinstance(entry['n'], int) and entry['n']>0
            assert isinstance(entry['edges'], int) and entry['edges']>=0
            assert isinstance(entry['k_chosen'], int) and entry['k_chosen']>=2
            break
    assert found, f'No index entry contains all required keys: {required}'
