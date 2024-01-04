from setuptools import find_packages, setuptools,setup
from typing import List 
HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function returns a list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requiments=file_obj.readlines()
        [req.replace("\n","")for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
name='mlprojectT',
version='0.0.1',
author='Amogh',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)