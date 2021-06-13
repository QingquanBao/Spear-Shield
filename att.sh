#!/bin/bash
python attack_main.py --step_size=0.01 --criterion=Margin+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=50
python attack_main.py --step_size=0.01 --criterion=Margin+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=80
python attack_main.py --step_size=0.01 --criterion=Margin+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=50 --random=0.01
python attack_main.py --step_size=0.01 --criterion=Margin+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=80 --random=0.01
python attack_main.py --step_size=0.01 --criterion=CE+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=50
python attack_main.py --step_size=0.01 --criterion=CE+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=80
python attack_main.py --step_size=0.01 --criterion=CE+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=50 --random=0.01
python attack_main.py --step_size=0.01 --criterion=CE+Entro --gpu_id=1,2 --model_name=model3  --msg=constBiasLoss --perturb_steps=80 --random=0.01
