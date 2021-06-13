#!/bin/bash
for model in  'model4' 'model5' 'model6' 'model1' 'model2' 'model3'
do
    for cri in 'CE' #'Margin+Entro' #'CE+Entro' 'CE' 'Margin'
    do
        for beta in 0.3 #0.5 0.7 1.0
        do
        python interesting.py --model_name=$model --beta=$beta --gpu_id=1
        done
    done
done
