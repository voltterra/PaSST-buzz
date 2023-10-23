TENSORBOARD_LOGDIR=/home/sagemaker-user/torch_runlogs/
pip install tensorboard
tensorboard --logdir=${TENSORBOARD_LOGDIR} &
