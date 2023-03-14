#!/bin/bash

VERSION=d0_v6
DATA_1=/home/data/kbh/DCASE2022_SELD_dataset/foa_dev/dev-test-sony
DATA_2=/home/data/kbh/DCASE2022_SELD_dataset/foa_dev/dev-test-tau
DATA_eval=/home/data/kbh/DCASE_eval/foa_eval/
OUT=/home/data/kbh/DCASE2022/out/${VERSION}
DEVICE=cuda:1
NP=1

python src/inference.py -c config/model/${VERSION}.yaml -m /home/nas/user/kbh/SELD/chkpt/${VERSION}/bestmodel.pt -i ${DATA_1} -o ${OUT} -n ${NP} -d ${DEVICE}
#python src/inference.py -c config/model/${VERSION}.yaml -m /home/nas/user/kbh/SELD/chkpt/${VERSION}/bestmodel.pt -i ${DATA_2} -o ${OUT} -n ${NP} -d ${DEVICE}
#python src/inference.py -c config/model/${VERSION}.yaml -m /home/nas/user/kbh/SELD/chkpt/${VERSION}/bestmodel.pt -i ${DATA_eval} -o ${OUT}_eval -n ${NP} -d ${DEVICE}
