package network;

public class utils {

	public static double transferDerivative(double output) {
		return output * (1.0-output);
	}
	
	
	static double sigmoid(double x) {
		return 1.0/(1+Math.exp(-x));
	}

	static double dotProduct(double[] weigths, double[] inputs) throws Exception {
		if (weigths.length != inputs.length) {
			throw new Exception("weigth and inputs vectors have different lengths.");
		}
		double res = 0;
		for (int i = 0; i < weigths.length; i++) {
			res += weigths[i]*inputs[i];
		}
		return res;
	}
}
