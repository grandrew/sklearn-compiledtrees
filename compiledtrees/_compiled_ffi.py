import types
import os.path
import sys
import importlib
from cffi import FFI


class CompiledClassifier:
    def __init__(self, cpp_f, func_name, outsz=2):
        # module = types.ModuleType("native_evaluate")
        # f_code = compile(open(cpp_f).read(), cpp_f, 'exec')
        # exec(f_code, module.__dict__)
        ffi = FFI()
        ffi.cdef(f"""
            int evaluate(float *f, double * probas);
        """)
        if type(cpp_f) != str:
            cpp_f = cpp_f.decode("utf8")
        C = ffi.dlopen(cpp_f)
        def eval_func(x_arr, out_arr=None):
            c_arr = ffi.new("float[]", x_arr)
            out_arr2 = ffi.new(f"double[{outsz}]")
            C.evaluate(c_arr, out_arr2)
            return out_arr2
        self.func = eval_func


        
    def predict_proba(self, x, output, num_samples=0):
        if num_samples == -1:
            return self.func(x, output)
        else:
            if num_samples == 0:
                num_samples = x.shape[0]
            for i in range(num_samples):
                self.func(x[i], output[i])
        # self.jit_cnt += 1
        # if self.jit_cnt % 50000 == 0:
            # print(pypyjit.get_stats_snapshot().counters)
        return output
