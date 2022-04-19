# Create the environment :
conda env create environment.yml -n env-rl
conda activate env-rl

cd jelly-bean-world/api/python
python setup.py install

conda install -c conda-forge gym==0.22.0 
conda install pytorch-lightning
conda install pytorch-lightning-bolts

conda env config vars set LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia-460:/usr/lib/nvidia
