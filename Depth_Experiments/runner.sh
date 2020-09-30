for depth in 6
do
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt SGD --depth $depth &
#	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt SWAG --depth $depth &
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt NA --depth $depth &
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt VOGN --depth $depth 
#	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt BBB --depth $depth &
done
