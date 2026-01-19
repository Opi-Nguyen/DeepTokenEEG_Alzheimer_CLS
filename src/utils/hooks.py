class FeatureCatcher:
    def __init__(self):
        self.cache = {}

    def hook_fn(self, name):
        def _fn(module, inp, out):
            self.cache[name] = out.detach().cpu()
        return _fn

    def clear(self):
        self.cache.clear()
