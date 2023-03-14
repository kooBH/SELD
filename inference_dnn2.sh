#!/bin/bash

VERSION=d0_v5
DATA_1=~/kiosk/DCASE2022_SELD_dataset/foa_dev/dev-test-sony
DATA_2=~/kiosk/DCASE2022_SELD_dataset/foa_dev/dev-test-tau
DATA_eval=~/kiosk/DCASE_eval/foa_eval/
DEVICE=cuda:1
NP=1


VERSION=d0_v6
OUT=/home/data2/kbh/DCASE2022/out/

python src/inference.py -c config/model/${VERSION}.yaml -m /home/nas/user/kbh/SELD/chkpt/${VERSION}/bestmodel.pt -i ${DATA_1} -o ${OUT} -n ${NP} -d ${DEVICE} -v ${VERSION}
python src/inference.py -c config/model/${VERSION}.yaml -m /home/nas/user/kbh/SELD/chkpt/${VERSION}/bestmodel.pt -i ${DATA_2} -o ${OUT} -n ${NP} -d ${DEVICE} -v ${VERSION}

#python src/inference.py -c config/model/${VERSION}.yaml -m /home/nas/user/kbh/SELD/chkpt/${VERSION}/bestmodel.pt -i ${DATA_eval} -o ${OUT} -n ${NP} -d ${DEVICE} -v ${VERSION}_eval
