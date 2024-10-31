deploy:
	pip install -e .

run-resnet:
	python3 examples/train_resnet.py --model_name='resnet18' --dataset='cifar10' --log_file='results/resnet18_cifar10' --world_size=1 --rank=0 --dist_url='tcp://192.168.1.169:4003' --hook='default' --pruning_amount=0.9

run-vit:
	python3 examples/train_vit.py --model_name='vit-base16' --dataset='cifar100' --log_file='results/vit-base16_cifar100' --world_size=1 --rank=0 --dist_url='tcp://192.168.1.169:4003' --hook='default' --pruning_amount=0.8

dist-run-resnet:
	torchrun --nnodes=$(world_size) --nproc_per_node=1 --node_rank=$(rank) --master_addr=192.168.1.154 --master_port=8003 examples/train_resnet.py --model_name=$(model_name) --dataset=$(dataset) --log_file='results/$(model_name)_$(dataset)_$(rank)' --rank=$(rank) --world_size=$(world_size) --dist_url='tcp://192.168.1.154:8003' --pruning_amount=$(pruning_amount) --hook=$(hook) --compression_ratio=$(compression_ratio) --threshold=$(threshold)

dist-run-vit:
	torchrun --nnodes=$(world_size) --nproc_per_node=1 --node_rank=$(rank) --master_addr=192.168.1.154 --master_port=8003 examples/train_vit.py --model_name='vit-base16' --dataset=$(dataset) --log_file='results/vit-base16_$(dataset)_$(rank)' --rank=$(rank) --world_size=$(world_size) --dist_url='tcp://192.168.1.154:8003' --pruning_amount=$(pruning_amount) --compression_ratio=$(compression_ratio)