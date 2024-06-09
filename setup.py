from setuptools import setup, Extension

setup_args = dict(
    ext_modules=[
        Extension(
            'voice100_pinktrombone._pinktrombone',
            ['csrc/capi.cpp', 'csrc/noise.cpp', 'csrc/pymain.c'],
            include_dirs=['include'],
            py_limited_api=True
        )
    ]
)
setup(**setup_args)
