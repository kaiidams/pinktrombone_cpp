from setuptools import setup, Extension

setup_args = dict(
    ext_modules = [
        Extension(
            'pinktrombone.mymodule',
            ['csrc/noise.cpp', 'csrc/pinktrombone.cpp'],
            include_dirs = ['include'],
            py_limited_api = True
        )
    ]
)
setup(**setup_args)
