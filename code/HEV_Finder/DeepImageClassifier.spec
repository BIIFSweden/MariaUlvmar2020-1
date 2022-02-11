# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import glob, os
#from PyInstaller import log as logging 
#from PyInstaller import compat
from os import listdir

#mkldir = compat.base_prefix + "/Library/bin/" 
#logger = logging.getLogger(__name__)
#logger.info("MKL installed as part of numpy, importing that! "+ compat.base_prefix+ "/Library/bin/" )
#binaries = [(mkldir + "/" + mkl, '.') for mkl in listdir(mkldir) if mkl.startswith('mkl_')]
#logger.info(binaries)

a = Analysis(['DeepImageClassifier.py'],
             pathex=['./'],
             binaries=[], #binaries,
             datas=[('maskrcnn_vessels/logs/', 'maskrcnn_vessels/logs/')],
             hiddenimports=["matplotlib","mpl_toolkits"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pydFiles = glob.glob("imagecodecs/*.pyd")
print (pydFiles)
for pydFile in pydFiles:
    a.datas += [("imagecodecs/" + os.path.basename(pydFile),pydFile, "DATA")]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          #a.binaries,
          #Tree('imagecodecs'),
          #a.zipfiles,
          #a.datas,
          strip=False,
          upx=True,
          name='DeepImageClassifier',
          debug=False,
          bootloader_ignore_signals=False,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               Tree('imagecodecs'),
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='DeepImageClassifier')
