from __future__ import annotations

from support.app import DEFAULT_WEB_PORT, app


if __name__ == "__main__":
    print(f"Starting Speech Recognition System at http://127.0.0.1:{DEFAULT_WEB_PORT}")
    app.run(host="127.0.0.1", port=DEFAULT_WEB_PORT, debug=False, use_reloader=False)
