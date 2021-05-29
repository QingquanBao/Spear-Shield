#!/bin/bash
for model in 'model1' 'model2' 'model3' 'model4' 'model5' 'model6'
do
    for cri in 'Margin+Entro' 'CE+Entro' 'CE' 'Margin'
    do
        for stepnum in 20 50 80
        do
        python attack_main.py --model_name=$model --criterion=$cri --batch_size=300 --gpu_id=1,2,0,3 --perturb_steps=$stepnum --step_size=0.01 --msg=NoODI
        done
    done
done
