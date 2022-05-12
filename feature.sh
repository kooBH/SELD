#!/bin/bash

TARGET=d0

# real data
for i in dev-train-sony dev-train-tau dev-test-sony dev-test-tau; do
	python src/preprocess.py -c config/feature/${TARGET}.yaml -a /home/data/kbh/DCASE2022_SELD_dataset/foa_dev/$i -m /home/data/kbh/DCASE2022_SELD_dataset/metadata_dev/$i -o /home/data/kbh/DCASE2022/
done

# synth data
python src/preprocess.py -c config/feature/${TARGET}.yaml -a /home/data/kbh/DCASE2022_SELD_synth_data/foa -m /home/data/kbh/DCASE2022_SELD_synth_data/metadata -o /home/data/kbh/DCASE2022/
