for width in 32 64 128 256 512 1024 2048
do
	#python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt SGD --width $width &
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt SWAG --width $width --gpu 0 &
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt NA --width $width  --gpu 1 &
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt VOGN --width $width  --gpu 2 &
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt BBB --width $width  --gpu 3 &
	python3 MNIST_runner.py --eps 0.0 --lam 0.0 --opt HMC --width $width  --gpu 4 
done
