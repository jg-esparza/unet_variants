# Models:
# model=<model1>,<model2>
# available: unet,resunet,transunet,swinunet

# Seeds:
# project.seed

python ./scripts/benchmark.py -m +benchmark=benchmark model=unet,resunet,swinunet project.seed=1,2,3
