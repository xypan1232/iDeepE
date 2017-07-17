from setuptools import setup

config = {
    'description': 'iDeepE: inferring protein-RNA binding sites and motifs using local and global CNNs',
    'download_url': 'https://github.com/xypan1232/iDeepE',
    'version': '0.1.0',
    'packages': ['iDeepE'],
    'setup_requires': [],
    'install_requires': ['numpy', 'scikit-learn'],
    'scripts': [],
    'name': 'iDeepE'
}

if __name__== '__main__':
    setup(**config)
