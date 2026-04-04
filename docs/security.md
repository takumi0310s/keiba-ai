# Security Incident Response Procedures

Last updated: 2026-04-04

---

## Credential Inventory

| Credential | Storage | Used By |
|-----------|---------|---------|
| NETKEIBA_COOKIE | `.env` (gitignored) | scrape_training.py, bulk_scrape_comments.py, refresh_cookie.py, scrape_premium_data.py, etc. |
| NETKEIBA_PASSWORD | `.env` (gitignored) | tools/refresh_cookie.py (Playwright auto-login) |
| NETKEIBA_LOGIN_ID | `.env` (gitignored) | tools/refresh_cookie.py |
| JRDB_ID | `.env` (gitignored) | tools/scrape_jrdb.py, tools/download_jrdb.py |
| JRDB_PASSWORD | `.env` (gitignored) | tools/scrape_jrdb.py, tools/download_jrdb.py |
| DISCORD_WEBHOOK_BETS | `.env` (gitignored) | tools/notify.py, tools/race_auto_notify.py |
| DISCORD_WEBHOOK_UPDATES | `.env` (gitignored) | tools/notify.py |
| DISCORD_WEBHOOK_URL | `.env` (gitignored) | tools/notify.py (fallback) |

---

## 1. Cookie Leak (NETKEIBA_COOKIE)

### Immediate Steps (within 5 minutes)

1. **Invalidate the cookie** -- Log out of netkeiba from any browser session. This invalidates the server-side session.
2. **Rotate the cookie** -- Log in again from the browser, copy the new cookie, and update `.env`:
   ```
   NETKEIBA_COOKIE=<new_cookie_value>
   ```
   Or run: `python tools/refresh_cookie.py`
3. **Check for unauthorized access** -- Log in to netkeiba and review account activity/login history if available.
4. **Scrub git history** (if committed) -- If the cookie was committed to git:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   git push origin --force --all
   ```
   Or use `git-filter-repo` (preferred):
   ```bash
   pip install git-filter-repo
   git filter-repo --invert-paths --path .env
   git push origin --force --all
   ```
5. **Notify GitHub** -- If pushed to a public repo, consider the cookie fully compromised regardless of scrubbing (GitHub caches may persist).

### Risk Assessment
- **Impact**: Read access to netkeiba premium data (race data, training times, speed indices). No financial transactions possible.
- **Severity**: LOW-MEDIUM. Cookie expires naturally (typically within days/weeks).

---

## 2. JRDB Credential Leak (JRDB_ID / JRDB_PASSWORD)

### Immediate Steps (within 5 minutes)

1. **Change the password** -- Log in to JRDB member portal (https://www.jrdb.com/) and change the password immediately.
2. **Update `.env`** with the new password:
   ```
   JRDB_ID=<your_id>
   JRDB_PASSWORD=<new_password>
   ```
3. **Scrub git history** (if committed) -- Same procedure as Cookie Leak step 4 above.
4. **Check download logs** -- Review `logs/` directory for unauthorized download activity.
5. **Contact JRDB support** -- If the credentials were exposed publicly, email JRDB support to report the incident and request account review.

### Risk Assessment
- **Impact**: Unauthorized download of JRDB data files (horse racing statistics). Potential TOS violation if credentials are shared.
- **Severity**: MEDIUM. JRDB is a paid subscription service; unauthorized use may result in account suspension.

---

## 3. netkeiba Credential Leak (NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD)

### Immediate Steps (within 5 minutes)

1. **Change the password** -- Log in to netkeiba (https://regist.netkeiba.com/) and change the password immediately.
2. **Update `.env`** with the new password:
   ```
   NETKEIBA_LOGIN_ID=<your_id>
   NETKEIBA_PASSWORD=<new_password>
   ```
3. **Regenerate cookie** -- After password change, the old cookie is invalidated. Run:
   ```bash
   python tools/refresh_cookie.py
   ```
4. **Scrub git history** (if committed) -- Same procedure as Cookie Leak step 4 above.
5. **Review account** -- Check netkeiba account settings for any unauthorized changes (email, payment info).
6. **Enable 2FA** -- If netkeiba supports two-factor authentication, enable it.

### Risk Assessment
- **Impact**: Full account access including premium content, saved preferences, and potentially stored payment information.
- **Severity**: HIGH. Account credentials grant persistent access until password is changed.

---

## 4. Discord Webhook Leak

### Immediate Steps

1. **Delete the webhook** -- Go to Discord Server Settings > Integrations > Webhooks, and delete the compromised webhook.
2. **Create a new webhook** -- Create a replacement webhook in the same channel.
3. **Update `.env`** with the new webhook URL.
4. **Scrub git history** if committed.

### Risk Assessment
- **Impact**: Attacker can post messages to your Discord channel. Cannot read messages or access other server features.
- **Severity**: LOW.

---

## Prevention Checklist

- [x] `.env` is listed in `.gitignore` (line 6)
- [x] `.env` is NOT tracked by git (verified: 0 tracked files)
- [x] `.env` was never committed to git history (verified: 0 commits)
- [x] All credentials are loaded via `os.environ.get()` or `.env` file parsing
- [x] No hardcoded credential values in source code
- [x] Log files contain no credential strings (verified: 0 matches)
- [ ] **WARNING**: `tools/scrape_jrdb.py:884` prints `JRDB_ID` value to stdout -- consider masking

---

## Periodic Security Audit

Run quarterly:

```bash
# Check .env is gitignored
grep -n '.env' .gitignore

# Verify .env not in git history
git log --all --full-history -- .env --oneline

# Search for credential values in history
git log --all -S "PASSWORD" --oneline
git log --all -S "COOKIE" --oneline

# Check logs for leaked credentials
grep -ri "password\|cookie" logs/ 2>/dev/null
```
