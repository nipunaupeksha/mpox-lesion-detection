from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_prefixed_env()
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app