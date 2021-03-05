from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from distutils import sysconfig

import contextlib
import os
import subprocess
import tempfile
from joblib import Parallel, delayed
import types

import platform

if platform.system() == 'Windows':
    CXX_COMPILER = os.environ['CXX'] if 'CXX' in os.environ else None
    delete_files = False
else:
    CXX_COMPILER = sysconfig.get_config_var('CXX')
    delete_files = True

EVALUATE_FN_NAME = "evaluate"


class CodeGenerator(object):
    def __init__(self):
        self._file = tempfile.NamedTemporaryFile(mode='w+b',
                                                 prefix='compiledtrees_',
                                                 suffix='.cpp',
                                                 delete=delete_files)
        self._indent = 0

    @property
    def file(self):
        self._file.flush()
        return self._file

    def write(self, line):
        self._file.write(("  " * self._indent + line + "\n").encode("ascii"))

    @contextlib.contextmanager
    def bracketed(self, preamble, postamble):
        assert self._indent >= 0
        self.write(preamble)
        self._indent += 1
        yield
        self._indent -= 1
        self.write(postamble)


def code_gen_regressor_tree(tree, evaluate_fn=None, gen=None):
    """
    Generates Py code representing the evaluation of a tree.

    Writes code similar to:
    ```
          def evaluate(f):
            if f[9] <= 0.175931170583:
              return 0.0
            else:
              return 1.0
    ```

    to the given CodeGenerator object.
    """
    if gen is None:
        gen = CodeGenerator()

    def recur(node):
        if tree.children_left[node] == -1:
            assert tree.value[node].size == 1
            gen.write("return {0}".format(tree.value[node].item()))
            return

        branch = "if f[{feature}] <= {threshold}f".format(
            feature=tree.feature[node],
            threshold=tree.threshold[node])
        with gen.bracketed(branch, ""):
            recur(tree.children_left[node])

        with gen.bracketed("else:", ""):
            recur(tree.children_right[node])

    fn_decl = "def {name}(f):".format(
        name=evaluate_fn)
    with gen.bracketed(fn_decl, ""):
        recur(0)
    return gen.file


def _gen_regressor_tree(i, tree):
    """
    Generates Py code for i'th tree.
    """
    name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
    gen_tree = CodeGenerator()
    return code_gen_regressor_tree(tree, name, gen_tree)


# def code_gen_ensemble_regressor(trees, individual_learner_weight, initial_value,
#                                 gen=None, n_jobs=1):
#     """
#     Writes code similar to:

#     ```
#     extern "C" {
#       __attribute__((__always_inline__)) double evaluate_partial_0(float* f) {
#         if (f[4] <= 0.662200987339) {
#           return 1.0;
#         }
#         else {
#           if (f[8] <= 0.804652512074) {
#             return 0.0;
#           }
#           else {
#             return 1.0;
#           }
#         }
#       }
#     }
#     extern "C" {
#       __attribute__((__always_inline__)) double evaluate_partial_1(float* f) {
#         if (f[4] <= 0.694428026676) {
#           return 1.0;
#         }
#         else {
#           if (f[7] <= 0.4402526021) {
#             return 1.0;
#           }
#           else {
#             return 0.0;
#           }
#         }
#       }
#     }

#     extern "C" {
#       double evaluate(float* f) {
#         double result = 0.0;
#         result += evaluate_partial_0(f) * 0.1;
#         result += evaluate_partial_1(f) * 0.1;
#         return result;
#       }
#     }
#     ```

#     to the given CodeGenerator object.
#     """

#     if gen is None:
#         gen = CodeGenerator()

#     tree_files = [_gen_regressor_tree(i, tree) for i, tree in enumerate(trees)]

#     with gen.bracketed('extern "C" {', "}"):
#         # add dummy definitions if you will compile in parallel
#         for i, tree in enumerate(trees):
#             name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
#             gen.write("double {name}(float* f);".format(name=name))

#         fn_decl = "double {name}(float* f) {{".format(name=EVALUATE_FN_NAME)
#         with gen.bracketed(fn_decl, "}"):
#             gen.write("double result = {0};".format(initial_value))
#             for i, _ in enumerate(trees):
#                 increment = "result += {name}_{index}(f) * {weight};".format(
#                     name=EVALUATE_FN_NAME,
#                     index=i,
#                     weight=individual_learner_weight)
#                 gen.write(increment)
#             gen.write("return result;")
#     return tree_files + [gen.file]


# classifier code goes below
def code_gen_classifier_tree(tree, evaluate_fn=EVALUATE_FN_NAME, gen=None, weight=1.):
    """
    Generates C code representing the evaluation of a tree.

    Writes code similar to:
    ```
          def evaluate(f, o): 
            if f[9] <= 0.175931170583:
              o[0] = 0
              o[1] = 0.7
            else:
              o[0] = 0.3
              o[1] = 0
    ```

    to the given CodeGenerator object.
    """
    if gen is None:
        gen = CodeGenerator()

    def recur(node):
        if tree.children_left[node] == -1:
            assert tree.value[node].shape[0] == 1
            n_leaf_samples = tree.value[node].sum()
            assert n_leaf_samples > 0
            for i, val in enumerate(tree.value[node][0]):
                gen.write("o[{i}] += {val}".format(i=i, val=float(val)*weight/n_leaf_samples))
            return

        branch = "if f[{feature}] <= {threshold}:".format(
            feature=tree.feature[node],
            threshold=tree.threshold[node])
        with gen.bracketed(branch, ""):
            recur(tree.children_left[node])

        with gen.bracketed("else:", ""):
            recur(tree.children_right[node])

    fn_decl = "def {name}(f, o):".format(
        name=evaluate_fn)
    with gen.bracketed(fn_decl, ""):
        recur(0)
    return gen.file


def _gen_classifier_tree(i, tree, weight):
    """
    Generates Py code for i'th tree.
    """
    name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
    gen_tree = CodeGenerator()
    return code_gen_classifier_tree(tree, name, gen_tree, weight)


def code_gen_ensemble_classifier(trees, individual_learner_weight, initial_value,
                                gen=None, n_jobs=1):
    """
    Writes code similar to:

    ```
      def evaluate(f, probas) {
        evaluate_partial_0(f, probas[0], 0.1)
        votes[evaluate_partial_1(f)] += 0.1
    ```

    to the given CodeGenerator object.
    """
    if gen is None:
        gen = CodeGenerator()

    tree_files = [_gen_classifier_tree(i, tree, individual_learner_weight) for i, tree in enumerate(trees)]

    fn_decl = "def {name}(f, probas):".format(name=EVALUATE_FN_NAME)
    with gen.bracketed(fn_decl, ""):
        for i, _ in enumerate(trees):
            increment = "{name}_{index}(f, probas)".format(
                name=EVALUATE_FN_NAME,
                index=i)
            gen.write(increment)
    return tree_files + [gen.file]


def _compile(cpp_f):
    return cpp_f


def compile_code_to_object(files, n_jobs=1):
    # if ther is a single file then create single element list
    # unicode for filename; name attribute for file-like objects
    if isinstance(files, str) or hasattr(files, 'name'):
        files = [files]

    o_files = (Parallel(n_jobs=n_jobs, backend='threading')
               (delayed(_compile)(f.name) for f in files))

    so_f = tempfile.NamedTemporaryFile(mode='w+b',
                                       prefix='compiledtrees_',
                                       suffix='.py',
                                       delete=delete_files)
    # link trees
    fd = open(so_f.name, "w+")
    fd.write("\n".join([open(f.name).read() for f in files]))
    fd.flush()
    fd.close()

    return so_f
