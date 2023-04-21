# just copy paste this in your docker terminal to emulate the gitlab runner
cd /app
rm -rf *
git clone https://gitlab.com/mihaicristianpirvu/video-representations-extractor -b dev
cd video-representations-extractor/
export VRE_ROOT=`pwd`
export VRE_WEIGHTS_DIR=$VRE_ROOT/resources/weights
export PYTHONPATH="$PYTHONPATH:$VRE_ROOT"
export PATH="$PATH:$VRE_ROOT/bin"
pip uninstall -y media-processing-lib
python3 -m pip install -r requirements.txt
# this is because we are unable to maintain a library properly
cd .ci/imgur/
