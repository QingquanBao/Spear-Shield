This is the project in Machine Learning SJTU

## Attack
Our attack is implemented in `pgd_attack.py` the `ODIAttack()` class.
To use it, we should in your `.py` write:

```
attack = ODIAttack(args.step_size, args.epsilon, args.perturb_steps)
x_adv = attack(model, x ,y)
```

## Defend
To train the model, run `python train.py`

To infer the current model, run `python infer.py --model_path=<The .pt>`
To attack it, run `python attack.py --model_path=<The .pt>`

**Attention**: the main dir should include `mymodel.py`, `utils.py`, `utils_awp.py` and `eval_model.py`
**Attention**: the main dir should include the dir `model` including WideResNet34 

## Others
- `normal_train.py` is the normal training of NN.
- `pgd_train.py` reserves some trials of the author
- `odi.py` is the official implementaion of ODI attack
- `pgd_attack.py` actually contains other author's trial, such as Many PGD
