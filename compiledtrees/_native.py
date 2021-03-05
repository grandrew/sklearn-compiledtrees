import types
import os.path
import sys
import importlib

import pypyjit

class CompiledClassifier:
    def __init__(self, cpp_f, func_name):
        # module = types.ModuleType("native_evaluate")
        # f_code = compile(open(cpp_f).read(), cpp_f, 'exec')
        # exec(f_code, module.__dict__)
        cpp_f = cpp_f.decode("utf8")
        print(cpp_f)
        sys.path.append(os.path.dirname(cpp_f))
        module = importlib.import_module(os.path.basename(cpp_f).split(".")[0])
        # self.func = module.__dict__[func_name.decode('utf-8')]
        self.func = module.evaluate
        # pypyjit.enable_debug()
        # self.jit_cnt = 0


        
    def predict_proba(self, x, output, num_samples=0):
        if num_samples == 0:
            num_samples = x.shape[0]
        for i in range(num_samples):
            self.func(x[i], output[i])
        # self.jit_cnt += 1
        # if self.jit_cnt % 50000 == 0:
            # print(pypyjit.get_stats_snapshot().counters)
        return output
