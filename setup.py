from setuptools import setup, find_packages

# Function to read the contents of the requirements.txt file
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='LeafletSC',
    version='0.2.11',
    author='Karin Isaev, Columbia University and NYGC',
    author_email='ki2255@cumc.columbia.edu', 
    description='Alternative splicing quantification in single cells with Leaflet',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
     url='https://github.com/daklab/Leaflet',  
    license='MIT',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    include_package_data=False, 
)
