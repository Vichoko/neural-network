package network;

public class NeuralLayer {
	int position;
	Neuron[] neurons;
	double[] pastOutputs;
	/** CONSTRUCTORES*/
	NeuralLayer(int n, double[] weigths, double bias) {
		neurons = new Neuron[n];// explicit declaration
		for (int i = 0; i < n; i++) {
			neurons[i] = new Neuron(bias, weigths);
		}
	}
	NeuralLayer(int neuronQuantity, int inputSize) {
		neurons = new Neuron[neuronQuantity];
		// todas las neuronas de una layer tienen la misma cantidad de pesos
		// bias recomendado entre 0 y 1. Pesos recomendados entre 0 y 1.
		for (int i = 0; i < neuronQuantity; i++) {
			double[] weigths = new double[inputSize];
			for (int j = 0; j < inputSize; j++) {
				weigths[j] = Math.random();
			}
			neurons[i] = new Neuron(Math.random(), weigths);
		}
	}
	/** GETTERS 
	 * @throws Exception */
	double[] getPastOutputs() throws Exception {
		if (pastOutputs == null) {
			throw new Exception("Layer have not been feed before calling getPastOutputs.");
		}
		return pastOutputs;
	}
	
	/** METODOS */
	int getOutputSize() {
		return neurons.length;
	}
	
	double[] synapsis(double[] inputs) throws Exception {
		// must have same size as weights
		if (inputs.length != neurons[0].getWeigths().length) {
			throw new Exception("inputs and weights have different sizes.");
		}
		
		this.pastOutputs = new double[neurons.length];
		for (int i = 0; i < pastOutputs.length; i++) {
			pastOutputs[i] = neurons[i].synapsis(inputs);
		}
		return pastOutputs;
		
	}

}
