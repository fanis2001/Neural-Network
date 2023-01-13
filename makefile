neuralNetwork: neuralNetwork.c
	gcc -O2 -o neuralNetwork neuralNetwork.c -lm


run: neuralNetwork
	./neuralNetwork

all: neuralNetwork
	./neuralNetwork

clean:
	rm neuralNetwork
