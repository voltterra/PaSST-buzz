# DS DL 2023 project focusing on biodiveristy

To make the notebooks funcitonal, you need to install a dependency repo: `hear21passt`.
Follow the command chain below:

```
cd src
pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.25#egg=hear21passt' 
```

This will create a hear21passt project inside src directory in editable mode. The whole
repo and import behavior relies on this feature, as it will automatically make 
subfolders with python code discoverable and importable.


