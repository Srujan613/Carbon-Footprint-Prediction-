import logging

from flask import Flask

from config import Config
from ml import ml_service
from routes.predict import predict_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

try:
    from flask_cors import CORS
    HAS_FLASK_CORS = True
except ImportError:
    HAS_FLASK_CORS = False
    logging.warning("flask-cors not installed — falling back to manual CORS headers. "
                     "Run `pip install -r requirements.txt` to use the real extension.")


def create_app():
    app = Flask(__name__)

    if HAS_FLASK_CORS:
        CORS(app, origins=Config.CORS_ORIGINS)
    else:
        @app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = Config.CORS_ORIGINS
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            return response

    ml_service.load_models()  # loads real models if present, else MOCK_MODE

    app.register_blueprint(predict_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
