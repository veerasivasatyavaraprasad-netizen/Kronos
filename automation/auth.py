"""
Login for the web UI.

Enabled when KRONOS_USERS is set ("alice:secret,bob:other"). Passwords are hashed in memory,
sessions are signed with KRONOS_SECRET_KEY, and repeated failures from one IP are throttled.
Without KRONOS_USERS the UI stays open (fine on localhost, not on the internet).
"""
import os
import secrets
import time
from collections import defaultdict, deque

from flask import jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

PUBLIC_ENDPOINTS = {"login", "logout", "healthz", "static"}
MAX_FAILURES, WINDOW_SECONDS = 5, 300


def parse_users(spec: str) -> dict:
    users = {}
    for pair in (spec or "").split(","):
        if ":" in pair:
            name, pw = pair.split(":", 1)
            if name.strip() and pw:
                users[name.strip()] = generate_password_hash(pw)
    return users


def init_auth(app) -> bool:
    users = parse_users(os.getenv("KRONOS_USERS", ""))
    secret = os.getenv("KRONOS_SECRET_KEY")
    if users and not secret:
        print("Warning: KRONOS_SECRET_KEY not set; using a random key (everyone is logged out on restart).")
    app.secret_key = secret or secrets.token_hex(32)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("KRONOS_SECURE_COOKIES") == "1",
    )
    failures = defaultdict(deque)

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if not users:
            return redirect("/")
        error = None
        if request.method == "POST":
            ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
            q = failures[ip]
            while q and q[0] < time.time() - WINDOW_SECONDS:
                q.popleft()
            name, pw = request.form.get("username", ""), request.form.get("password", "")
            if len(q) >= MAX_FAILURES:
                error = "Too many failed attempts. Try again in a few minutes."
            elif name in users and check_password_hash(users[name], pw):
                session.clear()
                session["user"] = name
                q.clear()
                nxt = request.args.get("next", "/")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else "/")
            else:
                q.append(time.time())
                error = "Wrong username or password."
        return render_template("login.html", error=error), (401 if error else 200)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login") if users else "/")

    @app.before_request
    def require_login():
        if not users or request.endpoint in PUBLIC_ENDPOINTS or session.get("user") in users:
            return None
        if request.path.startswith("/api/"):
            return jsonify({"error": "Login required"}), 401
        return redirect(url_for("login", next=request.path))

    @app.context_processor
    def inject_user():
        return {"current_user": session.get("user") if users else None}

    return bool(users)
