package util;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import network.NeuralNetwork;
/**
 * Utilidades varias para el funcionamiento de la red neuronal.
 * @author vichoko
 *
 */
public class utils {
	
	/**
	 * Genera metricas para una red neuronal, a partir de datos de prueba; las cuales son retornadas en un HashMap. 
	 * Las metricas disponibles se pueden obtener con las llaves:
	 * "tasa_aciertos", "tasa_desaciertos", "precision" y "recall".
	 * 
	 * @param net Red que se desea evaluar.
	 * @param input Datos de entrada de prueba.
	 * @param expectedOutput Datos de salida esperados para los datos de prueba.
	 * @param threshold Umbral para separar clases binarias (entre 0 y 1).
	 * @param verbose Valor booleano para activar las impresiones en consola.
	 * @return HashMap que contiene metricas, grabadas por su nombre clave.
	 * @throws Exception Si entrada y salida esperada tienen distintas salidas.
	 */
	public static HashMap<String, Double> binaryMetrics(NeuralNetwork net, double[][] input, double[][] expectedOutput, double threshold, boolean verbose) throws Exception {
		if (input.length != expectedOutput.length) {
			throw new Exception("utils.metrics :: input & expectedOutput have different sizes.");
		}
		// Se almacenan metricas en hasmap, por el nombre de la metrica.
		HashMap<String, Double> metricsData = new HashMap<>();
		
		/** Calcular tabla de aciertos
		 * 
		 * 
		 * verdaero = 1; falso = 0 */
		int verdaderosPositivos = 0;
		int verdaderosNegativos = 0;
		int falsosPositivos = 0;
		int falsosNegativos = 0;
		for (int i = 0; i < input.length; i++) {
			int[] realOutput = net.binaryPredict(input[i], threshold);
			// expected class is 0
			if (expectedOutput[i][0] == 0 && 0 == realOutput[0]) 
				verdaderosNegativos++;
			else if (expectedOutput[i][0] == 0 && 1 == realOutput[0]) 
				falsosNegativos++;
			else if (expectedOutput[i][0] == 1 && 1 == realOutput[0])
				verdaderosPositivos++;
			else if (expectedOutput[i][0] == 1 && 0 == realOutput[0])
				falsosPositivos++;
			else
				System.out.println("expected is "+expectedOutput[i][0]+" real is "+realOutput[0]);
		}
		
	
		double tasaAciertos = (verdaderosPositivos + verdaderosNegativos)*1.0/input.length;
		double tasaDesaciertos = (falsosPositivos + falsosNegativos)*1.0/input.length;
		
		metricsData.put("tasa_aciertos", tasaAciertos);
		metricsData.put("tasa_desaciertos", tasaDesaciertos);
		metricsData.put("precision", verdaderosPositivos*1.0/(verdaderosPositivos+falsosPositivos));
		metricsData.put("recall", verdaderosPositivos*1.0/(verdaderosPositivos+falsosNegativos));
		
		if (verbose) {
			System.out.println();
			System.out.println("Metricas de desempeño de la red neuronal: ");
			System.out.println("Numero de experimentos: " + input.length);
			System.out.println("Verdaderos Positivos: " + verdaderosPositivos);
			System.out.println("Verdaderos Negativos: " + verdaderosNegativos);
			System.out.println("Falsos Positivos: " + falsosPositivos);
			System.out.println("Falsos Negativos: " + falsosNegativos);
			System.out.println("Tasa aciertos: " + tasaAciertos + "; tasa desaciertos: " + tasaDesaciertos);
			System.out.println("Precision: " + metricsData.get("precision"));
			System.out.println("Recall: " + metricsData.get("recall"));
				
		}
		
		return metricsData;
	}
/**
 * Normaliza el input, para llevarlos a valores entre 0 y 1.
 * @param inputs vector de entrada que se desea normalizar
 * @param minVal valor minimo dentro del vector de entrada
 * @param maxVal valor maximo dentro del vector de entrada
 * @return arreglo de entradas con valores normalizados entre 0 y 1
 */
	public static double[][] normalize(double[][] inputs, double minVal, double maxVal) {
		double[][] res = new double[inputs.length][];
		int inputIndex = 0;
		for (double[] vector : inputs) {
			
			int vectorIndex = 0;
			double[] normalizedVector = new double[vector.length];
			for (double value : vector) {
				normalizedVector[vectorIndex++] = (value - minVal)/(maxVal - minVal);
			}	
			res[inputIndex++] = normalizedVector;
		}
		return res;
	}
	
	public static double[][] normalizeAny(double[][] in){
		double minVal = Double.MAX_VALUE;
		double maxVal = Double.MIN_VALUE;
		for (double[] X : in) {
	        for (double val : X) {
		        if (val > maxVal) {
		        	maxVal = val;
		        }
		        if (minVal > val) {
		        	minVal = val;
		        }	
	        }
		}		
		return normalize(in, minVal, maxVal);
	}
	/**
	 * Aplica operacion inversa a normalizacion.
	 * @param inputs 
	 * @param minVal
	 * @param maxVal
	 * @return
	 */
	public static double[][] denormalize(double[][] inputs, double minVal, double maxVal) {
		double[][] res = new double[inputs.length][];
		int inputIndex = 0;
		for (double[] vector : inputs) {
			double[] normalizedVector = new double[vector.length];
			int vectorIndex = 0;
			for (double value : vector){
				normalizedVector[vectorIndex++] = ((minVal - maxVal)*value - minVal)/(-1);
			}
			res[inputIndex++] = normalizedVector;
		}
		return res;
	}
	

	
	/** METODOS DEL PAQUETE NETWORK */
	public static double transferDerivative(double output) {
		return output * (1.0-output);
	}
	
	
	public static double sigmoid(double x) {
		return 1.0/(1+Math.exp(-x));
	}

	public static double dotProduct(double[] weigths, double[] inputs) throws Exception {
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
