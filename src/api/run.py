from gunicorn.app.base import Application
from gunicorn import util
from os import getenv

def is_debug():
    return getenv("DEBUG") is not None


class GunicornApp(Application):
    def __init__(self, options=None):
        self.options = options or {}
        if is_debug():
            self.options["reload"] = True
        self.options["worker_class"] = "uvicorn.workers.UvicornWorker"
        self.usage = None
        self.callable = None
        super().__init__()
        self.do_load_config()

    def init(self, *args):
        cfg = {}
        for k, v in self.options.items():
            if k.lower() in self.cfg.settings and v is not None:
                cfg[k.lower()] = v
        return cfg

    def load(self):
        return util.import_app(self.options.get("module_path"))


if __name__ == "__main__":
    GunicornApp({"bind": "0.0.0.0:8080", "module_path": "api.app:app"}).run()
