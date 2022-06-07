from setuptools import setup

package_name = 'Traveler_GUI'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/style.kv']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roboland',
    maintainer_email='qianlabusc@gmail.com',
    description='This is the package for traveler GUI',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['traveler_gui = Traveler_GUI.traveler_gui:main'
        ],
    },
)

