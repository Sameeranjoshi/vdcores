import os
import sys
import sysconfig

import torch
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))
generated_include_dir = os.path.join(this_dir, "build", "generated")
sources = [os.path.join(this_dir, "src", "torch_runtime.cu")]
runtime_obj = os.path.join(this_dir, "runtime.o")
include_dirs = [
    os.path.join(this_dir, "include"),
    os.path.join(this_dir, "include", "dae"),
    generated_include_dir,
]

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
extra_link_args = [f"-Wl,-rpath,{torch_lib}"]

# Project-local rpath for libamdhip64.so.6: torch+rocm ships an unversioned
# libamdhip64.so whose embedded SONAME is .so.6. The runtime loader looks for
# the SONAME exactly, so we add a build/hip_link_shim/ dir holding a symlink
# libamdhip64.so.6 -> torch's lib, and put it on rpath. (Created on demand
# inside the HIP build branch below.)
hip_link_shim = os.path.join(this_dir, "build", "hip_link_shim")

USE_HIP = os.environ.get("HIP", "0") == "1"

if USE_HIP:
    # AMD path: drive hipcc directly, bypassing torch.utils.cpp_extension which
    # only switches to hipcc when torch was built with ROCm. We compile and
    # link by hand and drop the resulting .so where build_ext.get_ext_fullpath
    # expects it.
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext as _build_ext

    HIPCC = os.environ.get("HIPCC", "hipcc")
    HIP_ARCH = os.environ.get("HIP_ARCH", "gfx942")

    # Refresh the libamdhip64.so.6 symlink so the runtime loader can resolve
    # torch's bundled HIP runtime (its file is unversioned but its SONAME is
    # .so.6). We rpath this dir into the .so so the lookup just works.
    os.makedirs(hip_link_shim, exist_ok=True)
    _torch_hip_lib = os.path.join(torch_lib, "libamdhip64.so")
    if os.path.exists(_torch_hip_lib):
        _shim = os.path.join(hip_link_shim, "libamdhip64.so.6")
        try:
            if os.path.islink(_shim) or os.path.exists(_shim):
                os.unlink(_shim)
        except OSError:
            pass
        os.symlink(_torch_hip_lib, _shim)

    try:
        from torch.utils.cpp_extension import include_paths as _torch_include_paths
        torch_includes = list(_torch_include_paths())
    except Exception:
        torch_includes = [
            os.path.join(os.path.dirname(torch.__file__), "include"),
            os.path.join(os.path.dirname(torch.__file__),
                         "include", "torch", "csrc", "api", "include"),
        ]

    py_include = sysconfig.get_paths()["include"]

    HIP_FLAGS = [
        f"--offload-arch={HIP_ARCH}",
        "-O3", "-std=c++20",
        "-D__HIP_PLATFORM_AMD__", "-D__AMDGCN_WAVEFRONT_SIZE=64",
        "-fPIC", "-DNDEBUG",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
    ]

    class HipBuildExt(_build_ext):
        def build_extension(self, ext):
            ext_path = self.get_ext_fullpath(ext.name)
            os.makedirs(os.path.dirname(ext_path), exist_ok=True)

            compile_flags = list(HIP_FLAGS) + [
                f"-DTORCH_EXTENSION_NAME={ext.name.rsplit('.', 1)[-1]}",
            ]
            includes = []
            for inc in (ext.include_dirs or []) + torch_includes + [py_include]:
                includes += ["-I", inc]

            objects = []
            for src in ext.sources:
                obj = os.path.join(
                    self.build_temp,
                    os.path.splitext(os.path.relpath(src, this_dir))[0] + ".o",
                )
                os.makedirs(os.path.dirname(obj), exist_ok=True)
                cmd = [HIPCC] + compile_flags + includes + [
                    "-c", src, "-o", obj,
                ]
                self.spawn(cmd)
                objects.append(obj)

            link_cmd = [HIPCC, "-shared", "-fPIC"] + objects
            link_cmd += list(ext.extra_objects or [])
            for libdir in ext.library_dirs or []:
                link_cmd += [f"-L{libdir}"]
            for lib in ext.libraries or []:
                link_cmd += [f"-l{lib}"]
            link_cmd += list(ext.extra_link_args or [])
            link_cmd += ["-o", ext_path]
            self.spawn(link_cmd)

    ext_modules = [
        Extension(
            name="dae.runtime",
            sources=sources,
            extra_objects=[runtime_obj],
            include_dirs=include_dirs,
            libraries=["torch", "torch_python", "c10", "amdhip64"],
            library_dirs=[torch_lib, hip_link_shim],
            extra_link_args=extra_link_args + [f"-Wl,-rpath,{hip_link_shim}"],
        )
    ]
    cmdclass = {"build_ext": HipBuildExt}

else:
    # NVIDIA / CUDA path: stock torch CUDAExtension.
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    ext_modules = [
        CUDAExtension(
            name="dae.runtime",
            sources=sources,
            extra_objects=[runtime_obj],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DNDEBUG"],
                "nvcc": [
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-Xptxas=-v",
                ],
            },
            libraries=["cuda"],
            extra_link_args=extra_link_args,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="dae",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
