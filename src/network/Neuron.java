package network;

import network.utils;

public class Neuron {
	public double getBias() {
		return bias;
	}

	public double[] getWeigths() {
		return this.weights;
	}

	
	double bias;
	double[] weights;
	double threshold = 0.5;
	double lastOutput;
	double delta;
	
	public Neuron(double bias, double[] weights){
		this.bias = bias;
		this.weights = weights;
	}
	
	public Neuron(double bias, double[] weigths, double threshold){
		this(bias, weigths);
		this.threshold = threshold;
	}

	
	
	public double synapsis(double[] inputs) throws Exception{
		lastOutput = utils.sigmoid(utils.dotProduct(this.weights, inputs) + bias); 
		return lastOutput;
	}
	
/** BINARY CLASSIFICATOR FEATURE*/
	public int binarySynapsis(double[] inputs) throws Exception{
		double val = synapsis(inputs);
		if (val > threshold)
			return 1;
		return 0;
	}
	
	
	enum E_feedType {POSITIVE, NEGATIVE};	
	void singleFeed(double learnRate, double[] inputs, E_feedType type) throws Exception {
		if (this.weights.length != inputs.length) {
			throw new Exception("weigth and inputs vectors have different lengths.");
		}
		
		if (type == E_feedType.POSITIVE) {
			for (int i = 0; i < this.weights.length; i++) {
				this.weights[i] += learnRate*inputs[i];
			}	
		} else {
			for (int i = 0; i < this.weights.length; i++) {
				this.weights[i] -= learnRate*inputs[i];
			}
		}
	}
	
	public void singleTrain(double learnRate, double[] inputs, int desiredOutput) throws Exception {
		if (desiredOutput<0 || desiredOutput>1) {
			throw new Exception("Output must be binary");
		}
		int output = this.binarySynapsis(inputs);
		if (output != desiredOutput) {
			if (desiredOutput == 0) {
				// Disminuir pesos
				singleFeed(learnRate, inputs, E_feedType.NEGATIVE);
				} else {
				// Aumentar pesos
				singleFeed(learnRate, inputs, E_feedType.POSITIVE);
			}
		}
		
		
	}

}
