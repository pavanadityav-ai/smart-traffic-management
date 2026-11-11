"""
Emergency alerts.

  Sound alert  always on: the dashboard plays a siren tone in the browser when
               the API reports a new alert.
  SMS alert    optional, through Twilio. It switches on when these environment
               variables are set (a free Twilio trial account works):
                 TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, ALERT_TO_NUMBER

A cooldown stops one ambulance from sending an alert on every frame.
SMS is sent in a background thread so video processing never waits for the network.
"""

import os
import threading

import requests

from traffic.config import ALERT_COOLDOWN_S


class TwilioSMS:
    API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

    def __init__(self, sid: str, token: str, from_number: str, to_number: str) -> None:
        self.sid, self.token, self.from_number, self.to_number = sid, token, from_number, to_number

    @classmethod
    def from_env(cls):
        keys = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "ALERT_TO_NUMBER")
        values = [os.environ.get(k) for k in keys]
        return cls(*values) if all(values) else None

    def send(self, text: str) -> str:
        try:
            response = requests.post(self.API.format(sid=self.sid), auth=(self.sid, self.token), timeout=10,
                                     data={"From": self.from_number, "To": self.to_number, "Body": text})
            return "sent" if response.status_code in (200, 201) else f"failed (HTTP {response.status_code})"
        except requests.RequestException as exc:
            return f"failed ({type(exc).__name__})"


class AlertManager:
    def __init__(self, sms: TwilioSMS | None = None, cooldown_s: float = ALERT_COOLDOWN_S) -> None:
        self.sms = sms
        self.cooldown_s = cooldown_s
        self._last_alert = None
        self.alerts: list[dict] = []

    @property
    def sms_enabled(self) -> bool:
        return self.sms is not None

    def check(self, emergency_types: list[str], now: float, location: str = "camera", on_sms_result=None):
        """Create an alert if an emergency vehicle is present and the cooldown has passed. Returns the alert or None."""
        if not emergency_types:
            return None
        if self._last_alert is not None and now - self._last_alert < self.cooldown_s:
            return None
        self._last_alert = now
        alert = {"id": len(self.alerts) + 1, "time_s": round(now, 1), "types": sorted(set(emergency_types)),
                 "sound": True, "sms": "queued" if self.sms else "disabled"}
        self.alerts.append(alert)
        if self.sms:
            text = f"Traffic alert: {', '.join(alert['types'])} detected at {location}. Signal switched to green."
            threading.Thread(target=self._send, args=(alert, text, on_sms_result), daemon=True).start()
        return alert

    def _send(self, alert: dict, text: str, on_sms_result) -> None:
        alert["sms"] = self.sms.send(text)
        if on_sms_result:
            on_sms_result(alert)
