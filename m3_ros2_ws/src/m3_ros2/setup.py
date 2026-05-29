from setuptools import find_packages, setup
import os

package_name = 'm3_ros2'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            # Construct the full local path
            local_path = os.path.join(path, filename)
            # Construct the destination path (share/package_name/path)
            install_path = os.path.join('share', package_name, path)
            paths.append((install_path, [local_path]))
    return paths

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *package_files('meshes'),
        *package_files('urdf'),
        *package_files('worlds'),
        *package_files('config'),
        *package_files('launch'),
        *package_files('maps'),
        *package_files('rviz'),
        *package_files('rl_policies'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pavan',
    maintainer_email='18pavansai@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
