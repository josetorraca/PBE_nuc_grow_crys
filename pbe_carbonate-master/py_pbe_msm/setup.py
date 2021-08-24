
# from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name="py_pbe_msm",
    version='0.0.1',
    description="Building block for the PSD simulation with the MSM",
    packages=["py_pbe_msm"],
    author="Caio Marcellos",
    license='BSD',
    install_requires=['numpy', 'numba'],
    extras_require = {
        'sundials_integration':  ["scikits.odes"]
    }
)
