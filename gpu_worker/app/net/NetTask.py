from celery import Task
from neuralnets.util.tools import load_net

class NetTask(Task):
    _net = None
    _model_path = None

    def set_model_path(self, model_path):
        if self._model_path != model_path:
            self._model_path = model_path
            self._net = load_net(self._model_path)

    @property
    def net(self):
        if self._net is None:
            if self.model_path is None:
                return None
            self._net = load_net(self._model_path)
        return self._net