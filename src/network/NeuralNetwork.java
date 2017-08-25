package network;

import java.util.ArrayList;

public class NeuralNetwork {
	ArrayList<NeuralLayer> layers = new ArrayList<>();
	boolean isComplete = false;
	double learningRate = 0.1	;
	/** CONSTRUCCION */
	public NeuralNetwork() {	
		// Red vacia, learning rate	
	}
	
	public NeuralNetwork(double learningRate) {	
		// Red vacia
		this.learningRate = learningRate;
	}

	public void newInputLayer(int inputSize, int numberOfNeurons) throws Exception {
		if (layers.size() > 0) {
			throw new Exception("Tried adding more than one input layer");
		}
		layers.add(new NeuralLayer(numberOfNeurons, inputSize));
	}
	
	public void newHiddenLayer(int numberOfNeurons) throws Exception {
		if (layers.size() == 0) {
			throw new Exception("Tried adding hidden layer without input layer");
		} else if (isComplete) {
			throw new Exception("Tried adding hidden layer after network is closed (i.e. output layer added)");
		}
		NeuralLayer previousLayer = layers.get(layers.size()-1);
		layers.add(new NeuralLayer(numberOfNeurons, previousLayer.getOutputSize()));
	}
	
	public void closeNetwork() throws Exception {
		/** Transforma la ultima capa en Output Layer y cierra red a modificaicones*/
		if (layers.size() == 0) {
			throw new Exception("Tried adding output layer without input layer");
		} else if (isComplete) {
			throw new Exception("Tried adding more than one output layer, network already closed");
		}
		isComplete = true;
	}
	/** METODOS PUBLICOS */
	public void train(double[][] input, double[][] expectedOutput, int nEpochs) throws Exception {
		// recibe dataset de entrenamiento; varios input con sus respectivos output
		if (input.length != expectedOutput.length) {
			throw new Exception("train :: dataset input and expectedOutput arrays have different lenghts.");
			
		}
		for (int epochIndex = 0; epochIndex < nEpochs; epochIndex++) {
			double sumError = 0;
			for (int dataIndex = 0; dataIndex < input.length; dataIndex++) {
				// entrenar sobre cada par de vectores input/output.
				double[] realOutput = this.forwardFeed(input[dataIndex]);
				if (realOutput.length != expectedOutput[dataIndex].length) {
					throw new Exception("train :: one of layers realOutput/expectedOutput have different sizes.");
					
				}

				for (int outputIndex = 0; outputIndex < realOutput.length; outputIndex++) {
					// Para cada input se calcula el error para visualizar aprendizaje
					sumError += Math.pow((expectedOutput[dataIndex][outputIndex]-realOutput[outputIndex]), 2);
				}
				this.backPropagation(expectedOutput[dataIndex]);
				this.updateWeights(input[dataIndex]);		
			}
			System.out.println("Epoch: "+epochIndex+", learnRate: "+learningRate+", error: "+sumError);	
		}
		}
	
	public int[] binaryPredict(double[] input, double threshold) throws Exception {
		int[] res = new int[forwardFeed(input).length];
		int index = 0;
		for (double i : forwardFeed(input)) {
			res[index++] = i > threshold ? 1 : 0;
		}
		return res;
	}
	
	public double[] predict(double[] input) throws Exception {
		return forwardFeed(input);
	}
	
	/** METODOS PRIVADOS */
	double[] forwardFeed(double[] food) throws Exception {
		for (NeuralLayer layer : layers) {
			food = layer.synapsis(food);
		}
		return food;
	}
	
	void backPropagation(double[] expectedOutput) {
		for (int layerIndex = layers.size() - 1; layerIndex > 0; layerIndex--) {
			// backward iteration
			NeuralLayer layer = layers.get(layerIndex);
			if (layerIndex == layers.size() - 1) {
				// Caso capa de salida
				int neuronIndex = 0;
				for (Neuron neuron : layer.neurons) {
					neuron.delta = (expectedOutput[neuronIndex++] - neuron.lastOutput)*
							utils.transferDerivative(neuron.lastOutput);
				}
			} else {
				// caso capa escondida o de entrada
				for (int neuronIndex = 0; neuronIndex < layer.neurons.length; neuronIndex++) {
					Neuron neuron = layer.neurons[neuronIndex];
					double error = 0;
					for (Neuron neighborNeuron : layers.get(layerIndex+1).neurons) {
						error += (neighborNeuron.weights[neuronIndex]*neighborNeuron.delta);
					}
					neuron.delta = error*utils.transferDerivative(neuron.lastOutput);
					
				}
				
			}
			
		}
	}
	
	void updateWeights(double[] input) throws Exception {
		for (int layerIndex = 0; layerIndex < this.layers.size(); layerIndex++) {
			NeuralLayer layer = layers.get(layerIndex);
			if (layerIndex > 0) {
				// Si no es input layer, los input vienen de layers previas
				input = layers.get(layerIndex-1).getPastOutputs();
			}
			for (Neuron neuron : layer.neurons) {
				// se actualiza su peso
				if (input.length != neuron.weights.length) {
					throw new Exception("updateWeights :: input and weight size incoherence");
				}
				for (int inputIndex = 0; inputIndex < input.length; inputIndex++) {
					neuron.weights[inputIndex] += this.learningRate*neuron.delta*input[inputIndex];
				}
				
			}
		}
	}
		

	
	
}
