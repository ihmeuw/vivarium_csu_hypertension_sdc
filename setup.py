#!/usr/bin/env python
import os

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_csu_hypertension_sdc", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'vivarium==0.9.3',
        'vivarium_public_health==0.10.4',
        'gbd_mapping==2.1.0',

        # These are pinned for internal dependencies on IHME libraries
        'numpy<=1.15.4',
        'tables<=3.4.0',
        'pandas<0.25',

        'scipy',
        'matplotlib',
        'seaborn',
        'jupyter',
        'jupyterlab',
        'pytest',
        'pytest-mock',
        'pyyaml',
        'click',
        'loguru',
        'jinja2',
    ]

    extras_require = [
        # For internal ihme use.
        'vivarium_cluster_tools==1.1.2',
        'vivarium_inputs[data]==3.1.1',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,
        extras_require={
            'dev': extras_require
        },

        # Includes data generating helpers for internal IHME use.
        entry_points='''
               [console_scripts]
               pcalculate_ckd_rr=vivarium_csu_hypertension_sdc.external_data.cli:pcalculate_ckd_rr [dev]
               pcalculate_proportion_hypertensive=vivarium_csu_hypertension_sdc.external_data.cli:pcalculate_proportion_hypertensive [dev]
               build_htn_artifact=vivarium_csu_hypertension_sdc.cli:build_htn_artifact [dev]
               make_specs=vivarium_csu_hypertension_sdc.cli:make_specs
        ''',
        # entry_points='''
        #          [console_scripts]
        #          make_specs=vivarium_csu_hypertension_sdc.cli:make_specs
        # ''',

        zip_safe=False,
    )
