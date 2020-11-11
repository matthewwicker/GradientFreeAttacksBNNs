for depth in 4 5 6 
do
	python3 FMNIST_runner.py --eps 0.0 --lam 0.0 --opt HMC --depth $depth --gpu 0 &
#	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt SGD --depth $depth &
	python3 FMNIST_runner.py --eps 0.0 --lam 0.0 --opt SWAG --depth $depth --gpu 1 &
	python3 FMNIST_runner.py --eps 0.0 --lam 0.0 --opt NA --depth $depth --gpu 2 &
	python3 FMNIST_runner.py --eps 0.0 --lam 0.0 --opt VOGN --depth $depth --gpu 3 & 
	python3 FMNIST_runner.py --eps 0.0 --lam 0.0 --opt BBB --depth $depth --gpu 4
done
