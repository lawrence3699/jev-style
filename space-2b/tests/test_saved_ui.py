import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import saved_ui


def test_archived_results_and_receipt_match_provenance():
    data = json.loads((ROOT / 'saved_examples.json').read_text())
    assert len(data['examples']) == 3
    for record in data['examples']:
        evidence = record['evidence']
        assert hashlib.sha256((ROOT / evidence['archived_csv']).read_bytes()).hexdigest() == evidence['cache_csv_sha256']
        assert hashlib.sha256((ROOT / evidence['archived_indices']).read_bytes()).hexdigest() == evidence['cache_indices_sha256']
        assert record['model']['revision'] == 'c9504148dae7f7a43d67a5782f12a015a4b4022c'
    assert hashlib.sha256((ROOT / 'examples/receipt.png').read_bytes()).hexdigest() == saved_ui.RECEIPT_SHA256


def test_cached_cards_are_native_static_and_separate_from_live():
    page = saved_ui.render_saved_examples('receipt_approval', expand=True)
    assert 'Saved examples · Not live responses.' in page
    assert 'separate from the input above' in page
    assert '<script' not in page and '<form' not in page and 'onclick=' not in page
    assert 'id="saved-receipt_approval" open' in page
    assert 'data:image/png;base64,' in page
    assert page.count('<meter ') == 8


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.1, 1.1, True])
def test_invalid_recorded_probability_rejected(bad):
    with pytest.raises(ValueError):
        saved_ui._probabilities({'Yes':bad, 'No':0}, 'receipt_approval')


def test_saved_text_escaped():
    assert saved_ui._text('<img src=x onerror=alert(1)>', 'test').startswith('&lt;img')
