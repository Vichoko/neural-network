package network;

import java.util.ArrayList;

import org.plot.Plot;

import util.utils;
/**
 * Red neuronal implementada mediante neuronas con funcion de activacion sigmoidea.
 * Las neuronas se inicializan con pesos y bias aleatorios entre 0 y 1.
 * 
 * @author vichoko
 *
 */
public class NeuralNetwork {
	ArrayList<NeuralLayer> layers = new ArrayList<>();
	boolean isComplete = false;
	double learningRate = 0.1	;
	/** CONSTRUCCION */
	public NeuralNetwork() {	
		// Red vacia, learning rate	fijo
	}
	/**
	 * Inicia red vacia con tasa de aprendizaje explicita.
	 * @param learningRate Tasa de aprendizaje de las neuronas
	 */
	public NeuralNetwork(double learningRate) {	
		// Red vacia
		this.learningRate = learningRate;
	}

	/** Crea capa de entrada
	 * @param inputSize Cantidad de entradas
	 * @param numberOfNeurons Cantidad de neuronas
	 * @throws Exception En caso de agregar mas de una capa de entrada
	 */
	public void newInputLayer(int inputSize, int numberOfNeurons) throws Exception {
		if (layers.size() > 0) {
			throw new Exception("Tried adding more than one input layer");
		}
		layers.add(new NeuralLayer(numberOfNeurons, inputSize));
	}
	/**
	 * Crea capa escondida o de salida (si se llama closeNetork despues)
	 * @param numberOfNeurons Numero de neuronas que tendra la capa
	 * @throws Exception
	 */
	public void newHiddenLayer(int numberOfNeurons) throws Exception {
		if (layers.size() == 0) {
			throw new Exception("Tried adding hidden layer without input layer");
		} else if (isComplete) {
			throw new Exception("Tried adding hidden layer after network is closed (i.e. output layer added)");
		}
		NeuralLayer previousLayer = layers.get(layers.size()-1);
		layers.add(new NeuralLayer(numberOfNeurons, previousLayer.getOutputSize()));
	}
	/**
	 * Transforma ultima capa oculta/entrada en capa de salida.
	 * Cierra la red neuronal a modificaciones topologicas.
	 * @throws Exception en caso de cerrar dos veces o si se llama antes de crear capa de entrada.
	 */
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
	/**
	 * Entrena la red neuronal con el conjunto de entrenamiento entregado en input y expectedOutput.
	 * La cantidad de elementos de input y expectedOutput deben coincidir.
	 *  
	 * @param input Entradas a la red neuronal, debe coincidir su cantidad con numero de entradas de la capa de entrada.
	 * @param expectedOutput Salidas esperadas de la red neuronal. Su cantidad debe coincidir con la cantidad de neuronas de la capa de salida.
	 * @param nEpochs Cantidad de veces que se entrenara con el data set entregado.
	 * @param errorPlotName if not null, error is plotted in file with this name.
	 * @throws Exception En caso de detectar inconsistencias entre input y expectedOutput.
	 */
	public void train(double[][] input, double[][] expectedOutput, int nEpochs, String errorPlotName) throws Exception {
		// recibe dataset de entrenamiento; varios input con sus respectivos output
		if (input.length != expectedOutput.length) {
			throw new Exception("train :: dataset input and expectedOutput arrays have different lenghts.");
			
		}
		System.out.println("learnRate: " + learningRate);
		double[] errors = new double[nEpochs];
		// Siguientes variables son para no saturar de impresiones en consola.
		int epochsPart = nEpochs/10;
		int counter = 0;
		int counter2 = 0;
		for (int epochIndex = 0; epochIndex < nEpochs; epochIndex++) {
			double sumError = 0;
			for (int dataIndex = 0; dataIndex < input.length; dataIndex++) {
				// entrenar sobre cada par de vectores input/output.
				double[] realOutput = this.forwardFeed(input[dataIndex]);
				if (realOutput.length != expectedOutput[dataIndex].length) {
					throw new Exception("train :: one of layers realOutput/expectedOutput have different sizes.");
					
				}

				for (int outputIndex = 0; outputIndex < realOutput.length; outputIndex++) {
					// Para cada input se calcula el error cuadratico medio para visualizar aprendizaje
					sumError += Math.pow((expectedOutput[dataIndex][outputIndex]-realOutput[outputIndex]), 2);
				}
				errors[epochIndex] = sumError;
				this.backPropagation(expectedOutput[dataIndex]);
				this.updateWeights(input[dataIndex]);		
			}
			if (counter2 == 0 || counter++ >= epochsPart) {
				System.out.println("Elapsed: " + (counter2++*10) + "Epoch: "+epochIndex+", error: "+sumError);
				counter = 0;
			}
		}
		
		// do grahphics
		if (errorPlotName != null) {
			double[] x = new double[errors.length];
			for (int i = 0; i < x.length; i++)
				x[i] = i+1;
			Plot plot = Plot.plot(Plot.plotOpts().title("Error vs Epochs")).
					xAxis("Epochs", null).
					yAxis("Error", null).
					series(null, Plot.data().xy(x, errors), null);
			plot.save(errorPlotName, "png");
			System.out.println("train :: 'Plot' de errores cuadraticos guardado en ../" + errorPlotName + ".png");
		}
		
		}
	/**
	 * Obtener prediccion de la red neuronal, dado una entrada particular.
	 * @param input Entrada que se desea hacer una prediccion.
	 * @return Salida predecida por la red neuronal. Valores entre 0 y 1. Su dimension coincide con la capa de salida.
	 * @throws Exception
	 */
	public double[] predict(double[] input) throws Exception {
		return forwardFeed(input);
	}
	
	/**
	 * Obtener prediccion de la red neuronal, obteniendo valores 0 o 1. Al pasar la prediccion real por umbral explicitado.
	 *
	 * @param input Entrada que se desea hacer una prediccion.
	 * @param threshold Umbral desde el cual se considerara la clase como 1. De lo contrario 0.
	 * @return Salida predecida, obteniendo valores 0 o 1.
	 * @throws Exception
	 */
	public int[] binaryPredict(double[] input, double threshold) throws Exception {
		int[] res = new int[forwardFeed(input).length];
		int index = 0;
		for (double i : forwardFeed(input)) {
			res[index++] = i > threshold ? 1 : 0;
		}
		return res;
	}
	
	
	/** METODOS (PRIVADOS) DE APRENDIZAJE MEDIANTE BACK PROPAGATION */
	

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
				neuron.bias += this.learningRate*neuron.delta;
			}
		}
	}
		

	
	
}
