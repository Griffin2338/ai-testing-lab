def test_case(case_id, results):
    r = results[case_id]
    assert r.status == "pass", f"{case_id} {r.title}: {r.summary} {' '.join(r.notes)}"
