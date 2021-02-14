import setuptools

setuptools.setup(
    name='graphical_models',
    version='0.1a.005',
    description='',
    long_description='',
    author='',
    author_email='',
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires='>3.5.0',
    zip_safe=False,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'numpy'
    ]
)

