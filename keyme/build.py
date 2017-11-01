import os
import subprocess

# subprocess.call("pyside-uic.exe keyme.ui -o keyme_gui.py")
# subprocess.call("pyside-uic.exe about.ui -o about_gui.py")
# subprocess.call('pyside-rcc.exe -py3 icons.qrc -o icons_rc.py')


subprocess.call("pyinstaller --onefile --clean --name=\"Key Detector\" --icon=key_icon.ico --windowed --uac-admin "
                "--hidden-import=timeit --hidden-import=bisect "
                "--additional-hooks-dir=C:\\Users\\ragha\\python_matlab\\python_and_matlab_codes\\keyme\\hooks\\ "
                "C:\\Users\\ragha\\python_matlab\\python_and_matlab_codes\\keyme\\keyme_main.py ")
