#!/bin/bash

#VERSION=d0_v0
#python ./src/train.py -c config/model/${VERSION}.yaml  -v ${VERSION} -d cuda:0

## 2022-05-24
#VERSION=d0_v0
#python ./src/train.py -c config/model/${VERSION}.yaml  -v ${VERSION} -d cuda:0 --chkpt /home/nas/user/kbh/SELD/chkpt/${VERSION}/bestmodel.pt -s 1344000

## 2022-05-26
#VERSION=d0_v0
#python ./src/train.py -c config/model/${VERSION}.yaml  -v ${VERSION} -d cuda:0

## 2022-05-29
#VERSION=d0_v2
#python ./src/train.py -c config/model/${VERSION}.yaml  -v ${VERSION} -d cuda:0

## 2022-06-09
VERSION=d0_v6
python ./src/train.py -c config/model/${VERSION}.yaml  -v ${VERSION} -d cuda:0
