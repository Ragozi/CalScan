"""Usage tracking and enforcement via Supabase."""

import logging
import os
from datetime import date, datetime, timezone
from typing import Literal

log = logging.getLogger("calscan")

FREE_SCANS_PER_MONTH = 8
UPGRADE_URL = "https://scan-to-cal-joy.lovable.app/checkout"

# Lazy-loaded Supabase client
_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is not None:
        return _supabase

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        log.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set — usage tracking disabled")
        return None

    try:
        from supabase import create_client
        _supabase = create_client(url, key)
        return _supabase
    except Exception as exc:
        log.error("Failed to init Supabase client: %s", exc)
        return None


def _is_new_month(last_reset: str | None) -> bool:
    """Return True if last_reset is from a previous month."""
    if not last_reset:
        return True
    try:
        reset_date = datetime.fromisoformat(last_reset).date()
        today = date.today()
        return reset_date.year != today.year or reset_date.month != today.month
    except Exception:
        return True


def get_usage(user_id: str) -> dict:
    """
    Fetch and return usage info for a user. Resets monthly count if needed.

    Returns:
        {
            "can_scan": bool,
            "scans_remaining": int | None,  # None = unlimited
            "tier": "free" | "pro",
            "message": str,
        }
    """
    sb = _get_supabase()
    if not sb:
        # Supabase not configured — allow scan, log warning
        log.warning("Usage check skipped — Supabase not configured")
        return {
            "can_scan": True,
            "scans_remaining": None,
            "tier": "unknown",
            "message": "Usage tracking unavailable",
        }

    try:
        # Fetch profile row
        resp = sb.table("profiles").select(
            "tier, scans_used_this_month, last_reset_date, subscription_status"
        ).eq("id", user_id).single().execute()

        profile = resp.data

        if not profile:
            log.warning("No profile found for user_id=%s", user_id)
            return {
                "can_scan": False,
                "scans_remaining": 0,
                "tier": "free",
                "message": "User profile not found.",
            }

        tier: Literal["free", "pro"] = profile.get("tier", "free")
        scans_used: int = profile.get("scans_used_this_month", 0) or 0
        last_reset: str | None = profile.get("last_reset_date")

        # Auto-reset on new month
        if _is_new_month(last_reset):
            today_iso = date.today().isoformat()
            sb.table("profiles").update({
                "scans_used_this_month": 0,
                "last_reset_date": today_iso,
            }).eq("id", user_id).execute()
            scans_used = 0
            log.info("Monthly reset applied for user_id=%s", user_id)

        # Pro users — unlimited
        if tier == "pro":
            return {
                "can_scan": True,
                "scans_remaining": None,
                "tier": "pro",
                "message": "Unlimited scans — Pro plan active.",
            }

        # Free tier
        remaining = max(0, FREE_SCANS_PER_MONTH - scans_used)
        can_scan = remaining > 0

        if can_scan:
            message = (
                f"You have {remaining} free scan{'s' if remaining != 1 else ''} left this month."
            )
        else:
            message = (
                "You've used your 8 free scans this month. "
                "Upgrade to Pro for unlimited scans!"
            )

        return {
            "can_scan": can_scan,
            "scans_remaining": remaining,
            "tier": "free",
            "message": message,
        }

    except Exception as exc:
        log.exception("Usage check failed for user_id=%s: %s", user_id, exc)
        # Fail open — don't block scans if Supabase is down
        return {
            "can_scan": True,
            "scans_remaining": None,
            "tier": "unknown",
            "message": "Usage check temporarily unavailable.",
        }


def increment_usage(user_id: str) -> None:
    """Increment scans_used_this_month for a free-tier user."""
    sb = _get_supabase()
    if not sb:
        return

    try:
        # Only increment for free tier users
        resp = sb.table("profiles").select("tier, scans_used_this_month").eq(
            "id", user_id
        ).single().execute()

        profile = resp.data
        if not profile or profile.get("tier") == "pro":
            return

        current = profile.get("scans_used_this_month", 0) or 0
        sb.table("profiles").update({
            "scans_used_this_month": current + 1,
        }).eq("id", user_id).execute()

        log.info("Incremented usage for user_id=%s → %d", user_id, current + 1)

    except Exception as exc:
        log.error("Failed to increment usage for user_id=%s: %s", user_id, exc)
