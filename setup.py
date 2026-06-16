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
    sources = [
        'csrc/python_api_hip.cpp',
        'csrc/kernels/mega_moe_fused_hip.cu',
        'csrc/kernels/mega_moe_baseline_hip.cu',
    ]
    build_include_dirs = [
        f'{accelerator_home}/include',
        project_path('deep_gemm', 'include'),
        project_path('third-party', 'fmt', 'include'),
    ]
    build_libraries = []
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
third_party_include_dirs = [
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
        large_opt_root = project_path('megamoe', 'dcu_megamoe_large_opt')
        large_opt_k1_hipcc_flags = hipcc_flags + [
            '-DNDEBUG',
            '-mllvm',
            '-enable-num-vgprs-768=true',
        ]
        modules.extend([
            CUDAExtension(
                name='megamoe.dcu_megamoe_large_opt.K1_fused.k1_fused_ext',
                sources=[
                    os.path.join(large_opt_root, 'K1_fused', 'k1_fused_ext.cu'),
                    os.path.join(large_opt_root, 'K1_fused', 'k1_v3_fused_ext.cu'),
                ],
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': large_opt_k1_hipcc_flags,
                },
            ),
            CUDAExtension(
                name='megamoe.dcu_megamoe_large_opt.K2_fused.k2_fused_ext',
                sources=[os.path.join(large_opt_root, 'K2_fused', 'k2_fused_ext.cu')],
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': hipcc_flags + ['-DNDEBUG', '-ffast-math'],
                },
            ),
            CUDAExtension(
                name='megamoe.dcu_megamoe_large_opt.K3_fused.k3_fused_ext',
                sources=[os.path.join(large_opt_root, 'K3_fused', 'k3_fused_ext.cu')],
                include_dirs=build_include_dirs,
                libraries=build_libraries,
                library_dirs=build_library_dirs,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': hipcc_flags + ['-DNDEBUG'],
                },
            ),
            CUDAExtension(
                name='megamoe.dcu_megamoe_large_opt.K3_fused.k3_v3_fused_ext',
                sources=[os.path.join(large_opt_root, 'K3_fused', 'k3_v3_fused_ext.cu')],
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
            'megamoe.dcu_megamoe_large_opt',
            'megamoe.dcu_megamoe_large_opt.K1_fused',
            'megamoe.dcu_megamoe_large_opt.K2_fused',
            'megamoe.dcu_megamoe_large_opt.K3_fused',
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
            'megamoe.dcu_megamoe_large_opt.K1_fused': ['*.cu', '*.cuh', '*.s', '*.co'],
            'megamoe.dcu_megamoe_large_opt.K2_fused': ['*.cu'],
            'megamoe.dcu_megamoe_large_opt.K3_fused': ['*.cu', '*.cuh', '*.s', '*.co'],
        })
    return data


LARGE_OPT_ASM_CODE_OBJECTS = [
    (
        project_path(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1.co',
        ),
        'K1_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K1_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.co',
        ),
        'K1_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE.co',
        ),
        'K3_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.co',
        ),
        'K3_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE.co',
        ),
        'K3_CLANG',
    ),
    (
        project_path(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_PACK5.s',
        ),
        os.path.join(
            'megamoe',
            'dcu_megamoe_large_opt',
            'K3_fused',
            'DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_PACK5.co',
        ),
        'K3_CLANG',
    ),
]


def _large_opt_asm_clang(env_name):
    clang = os.environ.get(env_name) or os.environ.get('MEGAMOE_DCU_AOT_CLANG')
    if clang:
        return clang
    dtk = os.environ.get('DTK_ROOT') or os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH') or accelerator_home
    candidate = os.path.join(dtk, 'aillvm', 'bin', 'clang') if dtk else ''
    return candidate if candidate and os.path.exists(candidate) else 'clang'


def build_large_opt_asm_code_objects(output_root, temp_root):
    if not (IS_HIP_EXTENSION and package_name == 'megamoe') or DG_SKIP_CUDA_BUILD:
        return
    for src, rel_dst, clang_env in LARGE_OPT_ASM_CODE_OBJECTS:
        if not os.path.exists(src):
            raise FileNotFoundError(f'large-opt asm source not found: {src}')
        dst = os.path.join(output_root, rel_dst)
        obj = os.path.join(temp_root, 'large_opt_asm', os.path.basename(rel_dst) + '.o')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.makedirs(os.path.dirname(obj), exist_ok=True)
        clang = _large_opt_asm_clang(clang_env)
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
                dst,
            ],
            check=True,
        )


class CustomBuildExt(BuildExtension):
    def run(self):
        super().run()
        output_root = current_dir if self.inplace else self.build_lib
        build_large_opt_asm_code_objects(output_root, self.build_temp)


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
        generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')
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

        include_sources = [(project_path('deep_gemm', 'include', 'deep_gemm'), 'deep_gemm')] \
            if IS_HIP_EXTENSION and package_name == 'megamoe' else []
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
