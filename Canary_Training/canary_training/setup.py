import setuptools


#long_description = open('README.md').read()


setuptools.setup(
    name='canary_training',
    version='0.1',
    author='broydj',
    author_email='broydj@amazon.com',
    description='',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    url='',
    #packages=setuptools.find_packages('src'),
    #package_dir={'': 'src'},
    packages=['canary_training'],
    package_data={"canary_training": ["instance_data_unnormalized.csv","instance_price_info.csv"]}
    #package_dat a= {"canary_training": "instance_price_info.csv"}
    #classifiers=[
    #    'Programming Language :: Python :: 3',
    #    'Operating System :: OS Independent',
    #],
    #python_requires='>=3',
)
