# Isolating and Leveraging Controllable and Noncontrollable Visual Dynamics in World Models 

## Get Started
Iso-Dream is implemented and tested on Ubuntu 18.04 with python == 3.7, PyTorch == 1.9.0:

1. Create an environment 
   ```
   conda create -n iso-env python=3.7
   conda activate iso-env
   ```   

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## DMC / CARLA
### For CARLA environments:

  1. Setup
     Download and setup CARLA 0.9.10
     ```
     chmod +x setup_carla.sh
     ./setup_carla.sh
     ```
     Add to your python path:
     ```
     export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI
     export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI/carla
     export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
     ```
     and merge the directories.

  2. Training
     Terminal 1:
     ```
     cd CARLA_0.9.10
     bash CarlaUE4.sh -fps 20 -opengl
     ```

     Terminal 2:
     ```
     cd dmc_carla_iso
     python dreamer.py --logdir log/iso_carla --action_step 20 --step 50 --kl_balance 0.8 --action_scale 0.001 --seed 9 --configs defaults carla
     ```

  3. Evaluation
     ```
     cd dmc_carla_iso
     python test.py --logdir test --action_step 20 --step 50 --kl_balance 0.8 --configs defaults carla
     ```

## BAIR / RoboNet
Train and test Iso-Dream on BAIR and RoboNet datasets. Also, install Tensorflow 2.1.0 for BAIR dataloader.

1. Download BAIR data. 
   ```
   wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
   ```

2. Train the model. You can use the following bash script to train the model. The learned model will be saved in the `--save_dir` folder.
  The generated future frames will be saved in the `--gen_frm_dir` folder. 
    ```
    cd bair_robonet_iso
    sh train_iso_model.sh
    ```

# Acknowledgement
We appreciate the following github repos where we borrow code from:

https://github.com/thuml/predrnn-pytorch

