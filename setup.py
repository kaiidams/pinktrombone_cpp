from setuptools import setup, Extension

setup_args = dict(
    ext_modules=[
        Extension(
            'voice100_pinktrombone._pinktrombone',
            ['lib/capi.cpp', 'lib/noise.cpp', 'lib/pymain.c'],
            include_dirs=['include'],
            py_limited_api=True
        )
    ]
)
setup(**setup_args)
