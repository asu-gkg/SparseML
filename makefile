deploy:
	pip install -e .

run-resnet18:
	python3 examples/resnet.py --model_name='resnet18' --dataset='imagenet100' --log_file='results/resnet18_imagenet100' --world_size=1 --rank=0 --dist_url='tcp://192.168.1.169:4003' --hook='default'