"""Shipped third-party files and generated records are what their provenance says they are."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_upstream_files_match_their_provenance():
    record = json.loads((UPSTREAM / 'PROVENANCE.json').read_text(encoding='utf-8'))
    listed = {f['path']: f for source in record['sources'] for f in source['files']}
    shipped = sorted(p.relative_to(UPSTREAM).as_posix() for p in UPSTREAM.rglob('*') if p.is_file() and p.name != 'PROVENANCE.json')
    assert shipped == sorted(listed)
    for path, f in listed.items():
        data = (UPSTREAM / path).read_bytes()
        assert len(data) == f['size_bytes'] and hashlib.sha256(data).hexdigest() == f['sha256'], path
        assert hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() == f['git_blob'], path
    for source in record['sources']:
        assert (UPSTREAM / source['directory'] / 'LICENSE').is_file()


def test_robot_records_name_their_sources():
    sources = {s['directory']: s for s in json.loads((UPSTREAM / 'PROVENANCE.json').read_text(encoding='utf-8'))['sources']}
    used = set()
    for name, expected in (('rbrobout_ur30', {'robotnik_description', 'ur_description'}),
                           ('ridgeback_ur10e', {'ridgeback_description', 'ur_description'}),
                           ('husky_a300_ur10e', {'clearpath_platform_description', 'ur_description'})):
        record = json.loads((ROOT / f'inputs/robots/{name}.json').read_text(encoding='utf-8'))
        provenance = record['provenance']
        assert sha256(ROOT / f'inputs/robots/{name}.urdf') == provenance['urdf_sha256']
        named = set(provenance) & set(sources)
        assert named == expected
        for directory in named:
            assert sources[directory]['commit'] in provenance[directory]
        assert record['declared']                                         # what is not upstream data is said to be so
        used |= named
    assert used == set(sources)                                           # nothing is shipped that no record uses


def test_kernel_hash_is_the_one_of_the_notices():
    assert sha256(ROOT / 'vendor/pacdm_original.py') == 'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de'
