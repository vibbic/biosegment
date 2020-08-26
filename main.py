from fastapi.middleware.wsgi import WSGIMiddleware
from biosegment_frontend.index import app as frontend_app
from biosegment_backend.main import app as backend_app

frontend_app.enable_dev_tools(debug=True)
backend_app.mount("/", WSGIMiddleware(frontend_app.server))