from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements from requirements.txt file
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='water_sensor_fault_detection',
    version='0.0.1',
    author='SIDDHI VINAYAK DASH',
    author_email='siddhudash21@gmail.com',
    description='A machine learning project for detecting faults in water sensors',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)