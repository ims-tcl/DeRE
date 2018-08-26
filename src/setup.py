from setuptools import setup, find_packages

setup(
    name='dere',
    version='0.1',
    description='Declarative Relation Extraction',
    url='http://www.ims.uni-stuttgart.de/forschung/ressourcen/werkzeuge/DeRE.en.html',
    author='Heike Adel, Laura Bostan, Sean Papay, Roman Klinger, Sebastian Pad\u00f3',
    author_email="dereproject@ims.uni-stuttgart.de",
    license='Apache 2',
    python_requires='>= 3.7', 
    packages=find_packages(),
    install_requires=[
        'mypy == 0.620',
        'mypy-extensions == 0.4.1',
        'click == 6.7',
        'scipy == 1.1.0',
        'scikit-learn == 0.19.2',
        'sklearn-crfsuite == 0.3.6',
        'nltk == 3.3',
        'networkx == 2.1',
        'spacy == 2.0.12'
    ],
    dependency_links=[
        'git://github.com/numpy/numpy-stubs.git'
    ],
    zip_safe=False,
    include_package_data=True,
)
