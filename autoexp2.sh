#!/bin/bash
for model in 'model3' #'model3' 'model4' 'model5' 'model6'
do
    for cri in 'CE' 'Margin'  #'Margin+Entro' #'CE+Entro' 'CE' 'Margin'
    do
        for stepnum in 50 80
        do
        python attack_main.py --model_name=$model --criterion=$cri --batch_size=300 --gpu_id=2,3 --perturb_steps=$stepnum --step_size=0.01 --msg=PGDwithRandom --random=0.01 
        done
    done
done
