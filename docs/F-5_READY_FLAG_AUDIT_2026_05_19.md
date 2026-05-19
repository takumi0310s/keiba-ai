# F-5 admin_verify_v2 ready flag audit + fix (2026-05-19)

## Root Cause

`tools/admin_verify_v2.py` line 200-204 (pre-fix):

```python
ready = (
    registered_count == 0  # "5/18 1830 想定: まだ未登録 = OK to proceed admin"
    and bats_exist == len(SCHTASKS)
    and py_compile_ok == len(set(e["py"] for e in SCHTASKS))
)
```

The flag `ready_for_5_22_admin` was designed to mean **"pre-conditions met to proceed with admin /Create"** — i.e., `registered_count == 0` (none registered yet). This was correct for 5/18 17:30 when the intent was "bats and py OK, ready to go do the admin registration."

After F-3 admin completed (9/9 registered), `registered_count == 9`, so `registered_count == 0` evaluates to `False`, and `ready_for_5_22_admin = False`. This is **logically correct** — but the message "ready for 5/22 admin: False" is **misleading** because it looks like "system is not ready" when in fact registration is complete.

**Classification: Case A** — flag meaning is "admin operation needed / pre-admin state". 9/9 = False is correct, but display is confusing.

## Fix Applied

Added three new fields to the `summary` dict:

| Field | Meaning | 9/9 registered | 0/9 unregistered |
|-------|---------|---------------|-----------------|
| `registration_complete` | schtasks 9/9 registered | `True` | `False` |
| `registration_status` | human-readable string | `"COMPLETE (9/9) no admin needed"` | `"NOT REGISTERED (0/9) run admin /Create"` |
| `fire_ready` | 5/23 fire READY = 9/9 + bat OK + py OK | `True` (if bat/py OK) | `False` |

Kept `ready_for_5_22_admin` for backward compatibility with its original meaning (pre-admin state indicator).

## Files Changed

- `tools/admin_verify_v2.py`: `run_verify()` — new flags; `print_summary()` — updated display
- `tests/test_admin_verify_v2.py`: 4 new test classes (F-5), 5 new test methods

## Test Results

13/13 passed (8 pre-existing + 5 new F-5 tests).

## State at 2026-05-19

Live run shows `schtasks registered: 0/9`. This indicates F-3 admin session registrations may not be visible in the current user context (admin tasks registered under a different elevation session). The tool correctly reports `fire_ready: False` and `registration_status: NOT REGISTERED`. Admin re-registration with `setup_all_tasks.bat` (admin elevation) required before 5/23 fire.
