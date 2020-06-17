# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import argparse

from bigdl.optim.optimizer import *
from zoo.pipeline.api.torch import TorchModel, TorchLoss
from zoo.pipeline.estimator import *
from zoo.common.nncontext import *
from zoo.feature.common import FeatureSet
from zoo.pipeline.api.keras.metrics import Accuracy

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR

import torch.nn as nn

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-path', type=str, default='./data', help='datapath')
    args = parser.parse_args()
    net = get_network(args, use_gpu=False)
    print(net)
    loss_function = nn.CrossEntropyLoss()

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        args.path,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    cifar100_test_loader = get_test_dataloader(
        args.path,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    # init on yarn when HADOOP_CONF_DIR and ZOO_CONDA_NAME is provided.
    if os.environ.get('HADOOP_CONF_DIR') is None:
        sc = init_spark_on_local(cores=1, conf={"spark.driver.memory": "20g"})
    else:
        num_executors = 2
        num_cores_per_executor = 4
        hadoop_conf_dir = os.environ.get('HADOOP_CONF_DIR')
        zoo_conda_name = os.environ.get('ZOO_CONDA_NAME')  # The name of the created conda-env
        sc = init_spark_on_yarn(
            hadoop_conf=hadoop_conf_dir,
            conda_name=zoo_conda_name,
            num_executor=num_executors,
            executor_cores=num_cores_per_executor,
            executor_memory="2g",
            driver_memory="10g",
            driver_cores=1,
            spark_conf={"spark.rpc.message.maxSize": "1024",
                        "spark.task.maxFailures":  "1",
                        "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})

    iter_per_epoch = len(cifar100_training_loader)
    warmup_delta = args.lr / (iter_per_epoch * args.warm)
    # iteration_per_epoch = int(math.ceil(float(len(cifar100_training_loader)) / args.b))
    zoo_lrSchedule = SequentialSchedule(iter_per_epoch)
    zoo_lrSchedule.add(Warmup(warmup_delta), iter_per_epoch * args.warm)
    zoo_lrSchedule.add(MultiStep([iter_per_epoch * 60, iter_per_epoch * 120, iter_per_epoch * 160], 0.2), iter_per_epoch * 200)
    zoo_optim = SGD(learningrate=warmup_delta, learningrate_decay=0.0, weightdecay=5e-4,
                momentum=0.9, dampening=0.0, nesterov=False,
                leaningrate_schedule=zoo_lrSchedule)

    zoo_model = TorchModel.from_pytorch(net)
    zoo_loss = TorchLoss.from_pytorch(loss_function)
    zoo_estimator = Estimator(zoo_model, optim_methods=zoo_optim)
    train_featureset = FeatureSet.pytorch_dataloader(cifar100_training_loader)
    test_featureset = FeatureSet.pytorch_dataloader(cifar100_test_loader)
    from bigdl.optim.optimizer import MaxEpoch, EveryEpoch
    zoo_estimator.train_minibatch(train_featureset, zoo_loss,
                                  end_trigger=MaxEpoch(settings.EPOCH),
                                  checkpoint_trigger=EveryEpoch(),
                                  validation_set=test_featureset,
                                  validation_method=[Accuracy()])
