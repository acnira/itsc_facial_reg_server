# ITSC Smart Lock System for Jetson
## TODO:
- [x] Modularize different parts
- [ ] Test and debug on the system
- [ ] Documentation
- [ ] Add Facial Recognition

## To setup the environment
## Server (folked from https://github.com/hkust-itsc/smart_door-jetson-.git)
1. Have python installed in your device (recommand python 3.8.8)
2. (OPTIONAL) Recommand using PyCharm (https://www.jetbrains.com/pycharm/) for IDE, and have Anaconda (https://www.anaconda.com/) installed for setup the environment.
3. Install the dependenies according to requirements.txt/freeze.yml using "pip install -r requirements.txt", use "conda env create -f freeze.yml" to setup the environment if you have Anaconda installed. If there are any failure, use "while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt" to install the dependencies line-by-line.
4. Download the RetinaFace folder from (https://gohkust.sharepoint.com/:f:/r/teams/ITSCTTInterns/Shared%20Documents/General/FRT/frtdev/home/py/insightface/RetinaFace?csf=1&web=1&e=RMnPQD) and extract to the corresponding locations("itsc_facial_reg_server/insightface/").
5. If you are using Windows, install a Linux VM for executing the following command, recommand using MSYS2(https://www.msys2.org/).
6. Go to  "itsc_facial_reg_server\insightface\RetinaFace" and run "make" with your linux command line, it may be failed as the insightface module(0.1.2) we are using is deprecated, try to update the module with latest version(https://github.com/deepinsight/insightface.git).
7. Open "facial_regconition_app.py" and set "demo = False" (L21) and uncommenting the import statement, then run this file.
8. You can test the server with running "test_facial_reg.py".

## Mobile
1. Have Node.js(https://nodejs.org/en/), React Native(https://reactnative.dev/), Android Studio(https://developer.android.com/studio?gclid=CjwKCAjw46CVBhB1EiwAgy6M4illQJ0pyIBnGHyh9d77GqKZe2dJUiBkGHJZFwLTWUc_Wt-Tr2uayxoCVqAQAvD_BwE&gclsrc=aw.ds) installed.
2. Run "npm install" for installing the project dependencies.
3. Create an android virtual device with using Android Studio, or connect a mobile device (Debug mode enabled) to your computer.
4. Run "adb devices" to see if your device is connected to the cmputer and ready for debug.
5. Remember to have your server running before debuging the app.
6. Run "npx react-native run-android"/"npx react-native run-ios" for running the app on your mobile device.