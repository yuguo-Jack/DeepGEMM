import ast
import os
import re
import shutil
import setuptools
import subprocess
import sys
import torch
import platform
import urllib
import urllib.error
import urllib.request
import warnings
from setuptools import find_packages
from setuptools.command.build_py import build_py
from packaging.version import parse
from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME, ROCM_HOME, IS_HIP_EXTENSION, BuildExtension
try:
    from setuptools.warnings import SetuptoolsDeprecationWarning
    warnings.filterwarnings(
        'ignore',
        message='setup.py install is deprecated.*',
        category=SetuptoolsDeprecationWarning)
except Exception:
    warnings.filterwarnings('ignore', message='setup.py install is deprecated.*')
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:
    _bdist_wheel = None
from scripts.generate_pyi import generate_pyi_file


DG_SKIP_CUDA_BUILD = int(os.getenv('DG_SKIP_CUDA_BUILD', '0')) == 1
DG_FORCE_BUILD = int(os.getenv('DG_FORCE_BUILD', '0')) == 1
DG_USE_LOCAL_VERSION = int(os.getenv('DG_USE_LOCAL_VERSION', '0' if IS_HIP_EXTENSION else '1')) == 1
DG_JIT_USE_RUNTIME_API = int(os.environ.get('DG_JIT_USE_RUNTIME_API', '0')) == 1

# Compiler flags
cxx_flags = ['-std=c++17', '-O3', '-fPIC', '-Wno-psabi', '-Wno-deprecated-declarations',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}']
if IS_HIP_EXTENSION:
    cxx_flags.append('-Wno-return-type')
if DG_JIT_USE_RUNTIME_API:
    cxx_flags.append('-DDG_JIT_USE_RUNTIME_API')
hipcc_flags = list(cxx_flags)
if IS_HIP_EXTENSION:
    hipcc_flags.append('--offload-arch=gfx938')

# Sources
current_dir = os.path.dirname(os.path.realpath(__file__))
project_path = lambda *parts: os.path.join(current_dir, *parts)
if IS_HIP_EXTENSION:
    package_name = os.getenv('DG_HIP_PACKAGE_NAME', 'megamoe')
    accelerator_home = ROCM_HOME or os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH') or '/opt/dtk'
    dcu_opt_root = project_path('megamoe', 'dcu_megamoe_opt')
    sources = [
        os.path.join(dcu_opt_root, 'csrc', 'python_api_hip.cpp'),
        os.path.join(dcu_opt_root, 'csrc', 'kernels', 'mega_moe_baseline_hip.cu'),
    ]
    build_include_dirs = [
        f'{accelerator_home}/include',
        os.path.join(dcu_opt_root, 'include'),
        os.path.join(dcu_opt_root, 'csrc'),
        project_path('third-party', 'fmt', 'include'),
    ]
    build_libraries = ['hsa-runtime64']
    build_library_dirs = [f'{accelerator_home}/lib', f'{accelerator_home}/lib64']
else:
    package_name = 'deep_gemm'
    accelerator_home = CUDA_HOME
    sources = ['csrc/python_api.cpp']
    build_include_dirs = [
        f'{CUDA_HOME}/include',
        f'{CUDA_HOME}/include/cccl',
        project_path('deep_gemm', 'include'),
        project_path('third-party', 'cutlass', 'include'),
        project_path('third-party', 'fmt', 'include'),
    ]
    build_libraries = ['cudart', 'nvrtc']
    build_library_dirs = [f'{CUDA_HOME}/lib64']
third_party_include_dirs = [] if IS_HIP_EXTENSION else [
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]

# Release
base_wheel_url = 'https://github.com/DeepSeek-AI/DeepGEMM/releases/download/{tag_name}/{wheel_name}'


def get_package_version():
    version_file = Path(current_dir) / package_name / '__init__.py'
    if not version_file.exists():
        version_file = Path(current_dir) / 'deep_gemm' / '__init__.py'
    with open(version_file, 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    revision = ''
    if DG_USE_LOCAL_VERSION:
        # noinspection PyBroadException
        try:
            status_cmd = ['git', 'status', '--porcelain']
            status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
            if status_output:
                revision = '+local'
            else:
                cmd = ['git', 'rev-parse', '--short', 'HEAD']
                revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except Exception:
            revision = '+local'
    return f'{public_version}{revision}'


def get_platform():
    if sys.platform.startswith('linux'):
        return f'linux_{platform.uname().machine}'
    else:
        raise ValueError('Unsupported platform: {}'.format(sys.platform))


def get_wheel_url():
    torch_version = parse(torch.__version__)
    torch_version = f'{torch_version.major}.{torch_version.minor}'
    python_version = f'cp{sys.version_info.major}{sys.version_info.minor}'
    platform_name = get_platform()
    deep_gemm_version = get_package_version()
    cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)

    # Determine the version numbers that will be used to determine the correct wheel
    # We're using the CUDA version used to build torch, not the one currently installed
    cuda_version = parse(torch.version.cuda)
    cuda_version = f'{cuda_version.major}'

    # Determine wheel URL based on CUDA version, torch version, python version and OS
    wheel_filename = f'deep_gemm-{deep_gemm_version}+cu{cuda_version}-torch{torch_version}-cxx11abi{cxx11_abi}-{python_version}-{platform_name}.whl'
    wheel_url = base_wheel_url.format(tag_name=f'v{deep_gemm_version}', wheel_name=wheel_filename)
    return wheel_url, wheel_filename


def get_ext_modules():
    if DG_SKIP_CUDA_BUILD:
        return []

    modules = [CUDAExtension(name=f'{package_name}._C',
                             sources=sources,
                             include_dirs=build_include_dirs,
                             libraries=build_libraries,
                             library_dirs=build_library_dirs,
                             extra_compile_args={'cxx': cxx_flags, 'nvcc': hipcc_flags})]
    if IS_HIP_EXTENSION and package_name == 'megamoe':
        opt_root = dcu_opt_root
        opt_k1_hipcc_flags = hipcc_flags + [
            '-DNDEBUG',
            '-mllvm',
            '-enable-num-vgprs-768=true',
        ]
        modules.extend([
            CUDAExtension(
                name='megamoe.dcu_megamoe_opt.K1_fused.k1_fused_ext',
                sources=[
                    os.path.join(opt_root, 'K1_fused', 'k1_fused_ext.cu'),
                    os.path.join(opt_root, 'K1_fused', 'k1_v3_fused_ext.cu'),
                ],
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': opt_k1_hipcc_flags,
                },
            ),
            CUDAExtension(
                name='megamoe.dcu_megamoe_opt.K2_fused.k2_fused_ext',
                sources=[os.path.join(opt_root, 'K2_fused', 'k2_fused_ext.cu')],
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': hipcc_flags + ['-DNDEBUG', '-ffast-math'],
                },
            ),
            CUDAExtension(
                name='megamoe.dcu_megamoe_opt.K3_fused.k3_fused_ext',
                sources=[os.path.join(opt_root, 'K3_fused', 'k3_fused_ext.cu')],
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': hipcc_flags + ['-DNDEBUG'],
                },
            ),
            CUDAExtension(
                name='megamoe.dcu_megamoe_opt.K3_fused.k3_v3_fused_ext',
                sources=[os.path.join(opt_root, 'K3_fused', 'k3_v3_fused_ext.cu')],
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': hipcc_flags + ['-DNDEBUG'],
                },
            ),
        ])
    return modules


def get_python_packages():
    if IS_HIP_EXTENSION and package_name == 'megamoe':
        return [
            'megamoe',
            'megamoe.dcu_megamoe_opt',
            'megamoe.dcu_megamoe_opt.K1_fused',
            'megamoe.dcu_megamoe_opt.K2_fused',
            'megamoe.dcu_megamoe_opt.K3_fused',
        ]
    return find_packages('.')


def get_package_data():
    data = {
        'deep_gemm': [
            'include/deep_gemm/**/*',
            'include/cute/**/*',
            'include/cutlass/**/*',
        ] if not IS_HIP_EXTENSION else [],
    }
    if IS_HIP_EXTENSION and package_name == 'megamoe':
        data.update({
            'megamoe.dcu_megamoe_opt': [
                'include/mega_moe_dcu/*.cuh',
                'csrc/*.cpp',
                'csrc/apis/*.hpp',
                'csrc/kernels/*.cu',
                'tests/*.py',
                'scripts/*.sh',
            ],
            'megamoe.dcu_megamoe_opt.K1_fused': ['*.cu', '*.cuh', '*.s', '*.co'],
            'megamoe.dcu_megamoe_opt.K2_fused': ['*.cu'],
            'megamoe.dcu_megamoe_opt.K3_fused': ['*.cu', '*.cuh', '*.s', '*.co'],
        })
    return data


OPT_ASM_CODE_OBJECTS = [
    (
        project_path(
            'megamoe',
            'dcu_megamoe_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.co',
        ),
        'K1_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_UNIFIED_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_UNIFIED_PACK5.co',
        ),
        'K1_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.co',
        ),
        'K3_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_UNIFIED_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_UNIFIED_PACK5.co',
        ),
        'K3_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_PACK5.co',
        ),
        'K3_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_UNIFIED_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_UNIFIED_PACK5.co',
        ),
        'K3_CLANG',
    ),
]


PREBUILT_CODE_OBJECTS = [
    (
        project_path(
            'megamoe',
            'dcu_megamoe_opt',
            'K1_fused',
            'prebuilt',
            'gfx938',
            'deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_opt',
            'K1_fused',
            'deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co',
        ),
    ),
]


def _existing_paths(paths):
    return [path for path in paths if os.path.exists(path)]


def _glob_project_paths(*parts):
    import glob
    return glob.glob(project_path(*parts))


def _megamoe_extension_header_dependencies(ext_name):
    if not (IS_HIP_EXTENSION and package_name == 'megamoe'):
        return []
    shared_headers = _glob_project_paths('megamoe', 'dcu_megamoe_opt', 'include', 'mega_moe_dcu', '*.cuh')
    api_headers = _glob_project_paths('megamoe', 'dcu_megamoe_opt', 'csrc', 'apis', '*.hpp')
    k1_headers = _glob_project_paths('megamoe', 'dcu_megamoe_opt', 'K1_fused', '*.cuh')
    k3_headers = _glob_project_paths('megamoe', 'dcu_megamoe_opt', 'K3_fused', '*.cuh')
    if ext_name == 'megamoe._C':
        return _existing_paths(shared_headers + api_headers)
    if ext_name == 'megamoe.dcu_megamoe_opt.K1_fused.k1_fused_ext':
        return _existing_paths(shared_headers + k1_headers)
    if ext_name in (
        'megamoe.dcu_megamoe_opt.K3_fused.k3_fused_ext',
        'megamoe.dcu_megamoe_opt.K3_fused.k3_v3_fused_ext',
    ):
        return _existing_paths(shared_headers + k3_headers)
    return []


def _object_path_for_source(build_temp, source):
    rel_source = os.path.relpath(source, current_dir)
    rel_object = os.path.splitext(rel_source)[0] + '.o'
    return os.path.join(build_temp, rel_object)


def _remove_objects_stale_against_headers(build_temp, extensions):
    if not (IS_HIP_EXTENSION and package_name == 'megamoe') or DG_SKIP_CUDA_BUILD:
        return
    for ext in extensions:
        header_deps = _megamoe_extension_header_dependencies(ext.name)
        if not header_deps:
            continue
        newest_header_mtime = max(os.path.getmtime(path) for path in header_deps)
        for source in ext.sources:
            obj = _object_path_for_source(build_temp, source)
            if os.path.exists(obj) and os.path.getmtime(obj) < newest_header_mtime:
                print(f'Removing stale object after header update: {obj}')
                os.remove(obj)


def _opt_asm_clang(env_name):
    clang = os.environ.get(env_name) or os.environ.get('MEGAMOE_DCU_AOT_CLANG')
    if clang:
        return clang
    dtk = os.environ.get('DTK_ROOT') or os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH') or accelerator_home
    candidate = os.path.join(dtk, 'aillvm', 'bin', 'clang') if dtk else ''
    return candidate if candidate and os.path.exists(candidate) else 'clang'


def _generated_file_current(path, dependency):
    return (
        os.path.exists(path) and
        os.path.getsize(path) > 0 and
        os.path.getmtime(path) >= os.path.getmtime(dependency)
    )


def _copy_file_if_needed(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if _generated_file_current(dst, src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def build_opt_asm_code_objects(output_root, temp_root):
    if not (IS_HIP_EXTENSION and package_name == 'megamoe') or DG_SKIP_CUDA_BUILD:
        return
    for src, rel_dst, clang_env in OPT_ASM_CODE_OBJECTS:
        if not os.path.exists(src):
            raise FileNotFoundError(f'opt asm source not found: {src}')
        cached_dst = project_path(rel_dst)
        dst = os.path.join(output_root, rel_dst)
        obj = os.path.join(temp_root, 'opt_asm', os.path.basename(rel_dst) + '.o')
        if not _generated_file_current(cached_dst, src):
            os.makedirs(os.path.dirname(cached_dst), exist_ok=True)
            os.makedirs(os.path.dirname(obj), exist_ok=True)
            clang = _opt_asm_clang(clang_env)
            print(f'Building opt asm code object: {cached_dst}')
            subprocess.run(
                [
                    clang,
                    '-x',
                    'assembler',
                    '-target',
                    'amdgcn-amd-amdhsa',
                    '-mcode-object-version=4',
                    '-mcpu=gfx938',
                    '-c',
                    '-o',
                    obj,
                    src,
                ],
                check=True,
            )
            subprocess.run(
                [
                    clang,
                    '-target',
                    'amdgcn-amd-amdhsa',
                    obj,
                    '-o',
                    cached_dst,
                ],
                check=True,
            )
        else:
            print(f'Skipping up-to-date opt asm code object: {cached_dst}')
        _copy_file_if_needed(cached_dst, dst)


def copy_prebuilt_code_objects(output_root):
    if not (IS_HIP_EXTENSION and package_name == 'megamoe') or DG_SKIP_CUDA_BUILD:
        return
    for src, rel_dst in PREBUILT_CODE_OBJECTS:
        if not os.path.exists(src):
            raise FileNotFoundError(f'prebuilt code object not found: {src}')
        dst = os.path.join(output_root, rel_dst)
        _copy_file_if_needed(src, dst)


class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        _remove_objects_stale_against_headers(self.build_temp, self.extensions)
        super().build_extensions()

    def run(self):
        super().run()
        output_root = current_dir if self.inplace else self.build_lib
        build_opt_asm_code_objects(output_root, self.build_temp)
        copy_prebuilt_code_objects(output_root)


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Second, make clusters' cache setting default into `envs.py`
        self.generate_default_envs()

        # Third, generate and copy .pyi file to build root directory
        self.generate_pyi_file()

        # Finally, run the regular build
        build_py.run(self)

    def generate_pyi_file(self):
        pyi_root = './megamoe/dcu_megamoe_opt/csrc' \
            if IS_HIP_EXTENSION and package_name == 'megamoe' else './csrc'
        generate_pyi_file(name='_C', root=pyi_root, output_dir='./stubs')
        pyi_source = os.path.join(current_dir, 'stubs', '_C.pyi')
        pyi_target = os.path.join(self.build_lib, package_name, '_C.pyi')

        if os.path.exists(pyi_source):
            print(f"Copying .pyi file from {pyi_source} to {pyi_target}")
            os.makedirs(os.path.dirname(pyi_target), exist_ok=True)
            shutil.copy2(pyi_source, pyi_target)
        else:
            print(f"Warning: .pyi file not found at {pyi_source}")

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in ('DG_JIT_CACHE_DIR', 'DG_JIT_PRINT_COMPILER_COMMAND', 'DG_JIT_CPP_STANDARD'):
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        os.makedirs(os.path.join(self.build_lib, package_name), exist_ok=True)
        with open(os.path.join(self.build_lib, package_name, 'envs.py'), 'w') as f:
            f.write(code)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(
            self.build_lib,
            package_name if IS_HIP_EXTENSION and package_name == 'megamoe' else 'deep_gemm',
            'include',
        )
        os.makedirs(build_include_dir, exist_ok=True)

        include_sources = []
        include_sources.extend((os.path.join(current_dir, d), d.split('/')[-1]) for d in third_party_include_dirs)

        # Copy JIT includes to the build directory.
        for src_dir, dirname in include_sources:
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


if _bdist_wheel is not None:
    class CachedWheelsCommand(_bdist_wheel):
        def run(self):
            if IS_HIP_EXTENSION or DG_FORCE_BUILD or DG_USE_LOCAL_VERSION:
                return super().run()

            wheel_url, wheel_filename = get_wheel_url()
            print(f'Try to download wheel from URL: {wheel_url}')
            # noinspection PyBroadException
            try:
                with urllib.request.urlopen(wheel_url, timeout=1) as response:
                    with open(wheel_filename, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)

                # Make the archive
                if not os.path.exists(self.dist_dir):
                    os.makedirs(self.dist_dir)
                impl_tag, abi_tag, plat_tag = self.get_tag()
                archive_basename = f'{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}'
                wheel_path = os.path.join(self.dist_dir, archive_basename + '.whl')
                os.rename(wheel_filename, wheel_path)
            except (urllib.error.HTTPError, urllib.error.URLError):
                print('Precompiled wheel not found. Building from source...')
                # If the wheel could not be downloaded, build from source
                super().run()


if __name__ == '__main__':
    # noinspection PyTypeChecker
    cmdclass = {
        'build_py': CustomBuildPy,
        'build_ext': CustomBuildExt,
    }
    if _bdist_wheel is not None:
        cmdclass['bdist_wheel'] = CachedWheelsCommand

    setuptools.setup(
        name=package_name,
        version=get_package_version(),
        packages=get_python_packages(),
        package_data=get_package_data(),
        ext_modules=get_ext_modules(),
        zip_safe=False,
        cmdclass=cmdclass,
    )
