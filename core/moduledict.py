from munch import Munch

class ModuleDict(Munch):
    def __init__(self, module, submodules = None, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.submodules = submodules
    def __call__(self,*args, **kwargs):
        kwargs_default = self.toDict()
        kwargs_input = kwargs
        kwargs_default.update(kwargs_input)
        kwargs_default.pop('module')
        kwargs_default.pop('submodules')
        return self.module(*args, **kwargs_default)

class ModuleDict2(Munch):
    def __init__(self, module, submodules = None, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self.submodules = submodules
    def __call__(self,*args, **kwargs):
        kwargs_default = self.toDict()
        if self.submodules is not None:
            kwargs_default.update({k:v() for k, v in self.submodules.items()})
        kwargs_input = kwargs
        kwargs_default.update(kwargs_input)
        kwargs_default.pop('module')
        kwargs_default.pop('submodules')
        return self.module(*args, **kwargs_default)
    
import importlib.util
import sys
import uuid
# For illustrative purposes.
import tokenize
def import_config_from_path(*paths):
    out = Munch()
    for path in paths:
        temp_name = str(uuid.uuid4())
        spec = importlib.util.spec_from_file_location(temp_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[temp_name] = module
        spec.loader.exec_module(module)
        print(module.__file__)
        print(module.__name__)
        print(dir(module))
        out.update(module.config)
    return out